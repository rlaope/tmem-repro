"""Experiment 1: does subtracting a recorded write erase it from a shared fast-weight LoRA?

Setup (TMEM fast weights: SVD-init LoRA, A frozen, only B trained, plain SGD):
  * A stream of synthetic writes w_0..w_{N-1}; each write = one fictional person with F facts.
  * Target write k. We record dB_k = B(after w_k) - B(before w_k).
  * Full run:   train w_0..w_{k+m} in order          -> B_full(m)
  * Subtract:   B_sub(m) = B_full(m) - dB_k
  * Oracle:     train the same stream with w_k removed -> B_orc(m)   (exact deletion = retrain)
  m = number of writes after k; we checkpoint several m in one pass.

Reported per checkpoint m:
  residual      ||B_sub - B_orc|| / ||dB_k||        (0 = subtraction equals retraining)
  nll_target    answer NLL on k's facts, direct + paraphrase, for full / sub / oracle / base
  nll_keep      answer NLL on the other writes' facts (should stay like oracle)
  nll_general   answer NLL on general-knowledge QA (side effects)
  relearn       NLL on k's facts after a few SGD steps on them, sub vs oracle
                (sub recovers much faster than oracle -> suppressed, not erased)

Output: one JSON line per (seed, m) in --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tmem.svd_lora import FastWeights, attach_svd_lora, state_diff, state_dist, state_norm  # noqa: E402
from tmem.synth import GENERAL, make_stream  # noqa: E402
from tmem.update import UpdateConfig, _collate, encode_qa, online_sft_ids  # noqa: E402

QA = Tuple[str, str]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def answer_nll(model, tok, pairs: Sequence[QA], max_len: int, bs: int = 32) -> float:
    """Mean per-token NLL of the answer given the question (lower = remembers better)."""
    if not pairs:
        return float("nan")
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ex = [encode_qa(tok, q, a, max_len) for q, a in pairs]
    tot, n = 0.0, 0
    for s in range(0, len(ex), bs):
        ids, lab, att = _collate(ex[s : s + bs], pad, device)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        lp = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum"
        )
        tot += float(lp)
        n += int((tgt != -100).sum())
    return tot / max(n, 1)


@torch.no_grad()
def answer_acc(model, tok, pairs: Sequence[QA], max_new: int = 12) -> float:
    """Greedy generation; 1 if the gold value appears in the output (case-insensitive)."""
    if not pairs:
        return float("nan")
    device = next(model.parameters()).device
    hit = 0
    for q, a in pairs:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        hit += int(a.lower() in text.lower())
    return hit / len(pairs)


def write_once(model, tok, fast: FastWeights, pairs: Sequence[QA], cfg: UpdateConfig) -> List[float]:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ex = [encode_qa(tok, q, a, cfg.max_len) for q, a in pairs]
    return online_sft_ids(model, fast, ex, pad, cfg)


def run_seed(model, tok, fast: FastWeights, args, seed: int, out_f) -> None:
    stream = make_stream(args.n_writes, args.facts_per_write, seed)
    k = args.target
    checkpoints = sorted({m for m in args.after if k + m < len(stream)})
    last = k + max(checkpoints)
    cfg = UpdateConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len, seed=seed)
    tgt_direct = stream[k].direct_pairs()
    tgt_para = stream[k].para_pairs()
    general = GENERAL[: args.n_general]

    # base model (B = 0) reference
    fast.reset()
    base = {
        "nll_target_direct": answer_nll(model, tok, tgt_direct, args.max_len),
        "nll_target_para": answer_nll(model, tok, tgt_para, args.max_len),
        "nll_general": answer_nll(model, tok, general, args.max_len),
    }

    # pass 1: full stream, record dB_k, checkpoint B_full(m)
    fast.reset()
    full_states: Dict[int, Dict[str, torch.Tensor]] = {}
    dB_k = None
    t0 = time.time()
    for i in range(last + 1):
        before = fast.state() if i == k else None
        write_once(model, tok, fast, stream[i].train_pairs(), cfg)
        if i == k:
            dB_k = state_diff(fast.state(), before)
        if i >= k and (i - k) in checkpoints:
            full_states[i - k] = fast.state()
    t_full = time.time() - t0

    # pass 2: oracle = same stream without w_k, checkpoint B_orc(m)
    fast.reset()
    orc_states: Dict[int, Dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for i in range(last + 1):
        if i != k:
            write_once(model, tok, fast, stream[i].train_pairs(), cfg)
        if i >= k and (i - k) in checkpoints:
            orc_states[i - k] = fast.state()
    t_orc = time.time() - t0

    dB_norm = state_norm(dB_k)
    for m in checkpoints:
        S_full, S_orc = full_states[m], orc_states[m]
        S_sub = {n: S_full[n] - dB_k[n] for n in S_full}
        keep_idx = [j for j in range(k + m + 1) if j != k]
        keep = [p for j in keep_idx for p in stream[j].direct_pairs()][: args.n_keep_eval]
        row = {
            "seed": seed, "k": k, "m": m, "n_writes_seen": k + m + 1,
            "dB_k_norm": dB_norm,
            "B_full_norm": state_norm(S_full),
            "residual_abs": state_dist(S_sub, S_orc),
            "residual_rel": state_dist(S_sub, S_orc) / max(dB_norm, 1e-12),
            "full_vs_orc_rel": state_dist(S_full, S_orc) / max(dB_norm, 1e-12),
            "base": base,
            "sec_full_pass": t_full, "sec_oracle_pass": t_orc,
        }
        for name, S in (("full", S_full), ("sub", S_sub), ("oracle", S_orc)):
            fast.load(S)
            r = {
                "nll_target_direct": answer_nll(model, tok, tgt_direct, args.max_len),
                "nll_target_para": answer_nll(model, tok, tgt_para, args.max_len),
                "nll_keep": answer_nll(model, tok, keep, args.max_len),
                "nll_general": answer_nll(model, tok, general, args.max_len),
            }
            if args.gen_eval:
                r["acc_target_direct"] = answer_acc(model, tok, tgt_direct)
                r["acc_target_para"] = answer_acc(model, tok, tgt_para)
                r["acc_keep"] = answer_acc(model, tok, keep[: args.n_gen_keep])
            if name in ("sub", "oracle") and args.relearn_steps > 0:
                rcfg = UpdateConfig(lr=args.lr, epochs=args.relearn_steps, batch_size=len(tgt_direct),
                                    max_len=args.max_len, seed=seed)
                # relearn on the DIRECT phrasing only, test on the held-out paraphrase
                write_once(model, tok, fast, tgt_direct, rcfg)
                r["relearn_nll_target_para"] = answer_nll(model, tok, tgt_para, args.max_len)
                r["relearn_nll_target_direct"] = answer_nll(model, tok, tgt_direct, args.max_len)
            row[name] = r
        fast.reset()
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        print(
            f"seed={seed} m={m:3d} resid_rel={row['residual_rel']:.4f} "
            f"tgt_para nll full/sub/orc/base={row['full']['nll_target_para']:.3f}/"
            f"{row['sub']['nll_target_para']:.3f}/{row['oracle']['nll_target_para']:.3f}/"
            f"{base['nll_target_para']:.3f} keep sub/orc={row['sub']['nll_keep']:.3f}/"
            f"{row['oracle']['nll_keep']:.3f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=4)
    ap.add_argument("--n-writes", type=int, default=65)
    ap.add_argument("--facts-per-write", type=int, default=4)
    ap.add_argument("--target", type=int, default=0, help="index k of the write to delete")
    ap.add_argument("--after", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-keep-eval", type=int, default=64)
    ap.add_argument("--n-general", type=int, default=20)
    ap.add_argument("--relearn-steps", type=int, default=2)
    ap.add_argument("--gen-eval", action="store_true", help="also run greedy generation accuracy (slower)")
    ap.add_argument("--n-gen-keep", type=int, default=16)
    ap.add_argument("--out", default="runs/deletion.jsonl")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    fast = attach_svd_lora(model, rank=args.rank, last_n_layers=args.last_n_layers)
    print(f"model={args.model} device={device} dtype={args.dtype} fast_params="
          f"{sum(p.numel() for p in fast.parameters())}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps({"config": vars(args)}) + "\n")
        for s in args.seeds:
            run_seed(model, tok, fast, args, s, f)


if __name__ == "__main__":
    main()
