"""Experiment 1: does subtracting a recorded write erase it from a shared fast-weight LoRA?

Setup (TMEM fast weights: SVD-init LoRA, A frozen, only B trained, plain SGD):
  * A stream of synthetic writes w_0..w_{N-1}; each write = one fictional person with F facts.
  * Target write k (default 16, so k has earlier memories in front of it).
    We record dB_k = B(after w_k) - B(before w_k).
  * Full run:   train w_0..w_{k+m} in order          -> B_full(m)
  * Subtract:   B_sub(m) = B_full(m) - dB_k
  * Oracle:     train the same stream with w_k removed -> B_orc(m)   (exact deletion = retrain)
  m = number of writes after k; several m are checkpointed in one pass.

Reported per (seed, k, m):
  residual_rel     ||B_sub - B_orc|| / ||dB_k||   (0 = subtraction equals retraining)
  alpha_star       <B_full - B_orc, dB_k> / ||dB_k||^2 (best single scale of dB_k to subtract)
  cos_resid_dB     cos(B_sub - B_orc, dB_k)
  per variant (full / sub / oracle, plus diagnostics):
    nll_target_{direct,para}   answer NLL on k's facts
    nll_keep_before            all writes before k (same set for every m)
    nll_keep_near / _after     writes k+1..k+4 / all writes after k (no truncation)
    nll_general                general-knowledge QA
    p_true_*                   binary streams: P(true option) vs the other option
    gen_*                      --gen-eval: greedy answers, answer-type counts, target texts
    relearn_*                  NLL after a few SGD steps on k's direct phrasing (sub vs oracle)
Diagnostic variants (NLL / p_true only):
    sub_a{alpha}   B_full - alpha * dB_k           (over-subtraction check)
    sub_astar      B_full - alpha_star * dB_k
    sub_random     B_full - R, R random with the same per-module norm as dB_k
    sub_decoy      B_full - dB_decoy, dB_decoy = an unrelated person trained at k's position
    sub_repeat     (--repeat-full) a second identical full run minus its dB_k: noise floor

Output: one JSON line per (seed, k, m) in --out; with --save-deltas, every per-write dB of both
passes is saved for scripts/analyze_deltas.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tmem.answers import classify_binary, classify_open  # noqa: E402
from tmem.svd_lora import FastWeights, attach_svd_lora, state_diff, state_dist, state_norm  # noqa: E402
from tmem.synth import GENERAL, GENERAL_KEYS, KINDS, REPLAY, Write, make_stream_with_decoy  # noqa: E402
from tmem.update import UpdateConfig, _collate, encode_qa, online_sft_ids  # noqa: E402

QA = Tuple[str, str]
State = Dict[str, torch.Tensor]
Example = Tuple[List[int], List[int]]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ state algebra

def state_dot(a: State, b: State) -> float:
    return float(sum((a[k].float().cpu() * b[k].float().cpu()).sum() for k in a))


def state_axpy(x: State, y: State, alpha: float) -> State:
    """x - alpha * y, on x's device."""
    return {n: x[n] - alpha * y[n].to(x[n].device) for n in x}


def random_like(ref: State, seed: int) -> State:
    """Gaussian state with the same per-module Frobenius norm as ref."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, v in ref.items():
        r = torch.randn(v.shape, generator=g)
        out[n] = r * (v.float().norm() / r.norm().clamp_min(1e-12))
    return out


# ------------------------------------------------------------------ training passes

def encode_pairs(tok, pairs: Sequence[QA], max_len: int) -> List[Example]:
    return [encode_qa(tok, q, a, max_len) for q, a in pairs]


def encode_scored(tok, q: str, prefix: str, target: str, max_len: int) -> Example:
    """Chat prompt + answer prefix (no loss) + target (loss). No EOS: only the value is scored."""
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt + prefix, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    return (p_ids + t_ids)[:max_len], ([-100] * len(p_ids) + t_ids)[:max_len]


def rehearsal_pairs(earlier: Sequence[Write], seed: int, index: int, n: int) -> List[QA]:
    """n QA pairs from earlier writes' training pairs (an agent replaying its QA log); depends on (seed, index)."""
    pool = [p for w in earlier for p in w.train_pairs()]
    if n <= 0 or not pool:
        return []
    return random.Random(seed * 7_000_003 + index).sample(pool, min(n, len(pool)))


def replay_pairs(seed: int, index: int, n: int) -> List[QA]:
    """General QA mixed into write `index`; depends only on (seed, index), so full and oracle passes match."""
    if n <= 0:
        return []
    return random.Random(seed * 100_003 + index).sample(REPLAY, min(n, len(REPLAY)))


def train(model, fast: FastWeights, ex: Sequence[Example], pad: int, cfg: UpdateConfig) -> None:
    online_sft_ids(model, fast, ex, pad, cfg)


def full_pass(model, fast, enc: List[List[Example]], pad: int, cfg: UpdateConfig, k: int, last: int,
              checkpoints: Sequence[int], decoy: Optional[List[Example]] = None):
    """Train writes 0..last in order. Returns (states at k+m, per-write deltas, dB_decoy).

    The decoy (if given) is trained from the state right before k, its delta recorded, and the
    state restored, so the main stream is unchanged.
    """
    fast.reset()
    states: Dict[int, State] = {}
    deltas: Dict[int, State] = {}
    dB_decoy = None
    for i in range(last + 1):
        before = fast.state()
        if i == k and decoy is not None:
            train(model, fast, decoy, pad, cfg)
            dB_decoy = state_diff(fast.state(), before)
            fast.load(before)
        train(model, fast, enc[i], pad, cfg)
        deltas[i] = state_diff(fast.state(), before)
        if i >= k and (i - k) in checkpoints:
            states[i - k] = fast.state()
    return states, deltas, dB_decoy


def oracle_pass(model, fast, enc: List[List[Example]], pad: int, cfg: UpdateConfig, k: int, last: int,
                checkpoints: Sequence[int]):
    """Same stream without write k. Returns (states at k+m, per-write deltas for i > k)."""
    fast.reset()
    states: Dict[int, State] = {}
    deltas: Dict[int, State] = {}
    for i in range(last + 1):
        if i != k:
            before = fast.state() if i > k else None
            train(model, fast, enc[i], pad, cfg)
            if i > k:
                deltas[i] = state_diff(fast.state(), before)
        if i >= k and (i - k) in checkpoints:
            states[i - k] = fast.state()
    return states, deltas


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def answer_logprobs(model, tok, items: Sequence[Tuple[str, str, str]], max_len: int,
                    bs: int = 32) -> List[Tuple[float, int]]:
    """Per (question, prefix, target): (sum of target-token log-probs, number of target tokens)."""
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ex = [encode_scored(tok, q, pre, t, max_len) for q, pre, t in items]
    out: List[Tuple[float, int]] = []
    for s in range(0, len(ex), bs):
        ids, lab, att = _collate(ex[s: s + bs], pad, device)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        lp = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
        mask = tgt != -100
        for row_lp, row_m in zip((lp * mask).sum(1).tolist(), mask.sum(1).tolist()):
            out.append((row_lp, int(row_m)))
    return out


def answer_nll(model, tok, items: Sequence[Tuple[str, str, str]], max_len: int) -> float:
    """Mean per-token NLL of the target given question + prefix (lower = remembers better)."""
    if not items:
        return float("nan")
    lp = answer_logprobs(model, tok, items, max_len)
    return -sum(x for x, _ in lp) / max(sum(n for _, n in lp), 1)


def p_true(model, tok, items: Sequence[Tuple[str, str, str, str]], max_len: int) -> float:
    """Mean P(true option) when the model must pick between the two options (sequence log-probs)."""
    if not items:
        return float("nan")
    lt = answer_logprobs(model, tok, [(q, pre, t) for q, pre, t, _ in items], max_len)
    lw = answer_logprobs(model, tok, [(q, pre, w) for q, pre, _, w in items], max_len)
    return sum(1.0 / (1.0 + math.exp(b[0] - a[0])) for a, b in zip(lt, lw)) / len(items)


@torch.no_grad()
def generate(model, tok, questions: Sequence[str], max_new: int = 16, bs: int = 32) -> List[str]:
    """Greedy answers, batched with left padding."""
    if not questions:
        return []
    device = next(model.parameters()).device
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs: List[str] = []
    try:
        for s in range(0, len(questions), bs):
            prompts = [
                tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
                for q in questions[s: s + bs]
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        tok.padding_side = side
    return [o.strip() for o in outs]


def gen_eval(model, tok, binary: bool, items: Sequence[Tuple[str, str, str, str]], other_values: Sequence[str],
             person_values: Dict[str, List[str]], keep_texts: bool, max_new: int):
    """items = (question, gold, wrong-or-'', entity-or-''). Returns {acc, types, [texts]}."""
    if not items:
        return {"acc": float("nan"), "types": {}}
    texts = generate(model, tok, [q for q, _, _, _ in items], max_new=max_new)
    types = []
    for (q, gold, wrong, ent), t in zip(items, texts):
        types.append(classify_binary(t, gold, wrong, question=q) if binary
                     else classify_open(t, gold, other_values, person_values.get(ent, [])))
    hit = "true" if binary else "gold"
    r = {"acc": sum(x == hit for x in types) / max(len(types), 1), "types": dict(Counter(types))}
    if keep_texts:
        r["texts"] = [{"q": q, "gold": g, "out": t, "type": ty} for (q, g, _, _), t, ty in zip(items, texts, types)]
    return r


# ------------------------------------------------------------------ one (seed, k)

def facts_items(writes: Sequence[Write], para: bool = False) -> List[Tuple[str, str, str, str]]:
    out = []
    for w in writes:
        for f in w.facts:
            out.append(((f.para_q if para else f.train_qs[0]), f.value, f.wrong or "", w.entity))
    return out


def run_one(model, tok, fast: FastWeights, args, seed: int, k: int, out_f) -> None:
    binary = args.kind == "binary_ko"
    skw = dict(values=args.values, answer_format=args.answer_format, n_phrasings=args.n_phrasings)
    stream, decoy_w = make_stream_with_decoy(args.n_writes, args.facts_per_write, seed, args.kind, **skw)
    if not args.diag:
        decoy_w = None
    checkpoints = sorted({m for m in args.after if k + m < len(stream)})
    if not checkpoints:
        print(f"skip seed={seed} k={k}: no checkpoint fits in {len(stream)} writes", flush=True)
        return
    last = k + max(checkpoints)
    cfg = UpdateConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len, seed=seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # With --rehearse, each write also retrains pairs from the QA log of earlier writes.
    # Full run: the log contains k. Oracle: k was never written, so it is never in the log.
    # Deleting = subtract dB_k and drop k from the log; nothing is trained after the deletion here,
    # so the second part only matters through the oracle's log.
    def write_input(i: int, log: Sequence[Write]) -> List[Example]:
        return encode_pairs(tok, stream[i].train_pairs() + replay_pairs(seed, i, args.replay)
                            + rehearsal_pairs(log, seed, i, args.rehearse), args.max_len)

    enc = [write_input(i, stream[:i]) for i in range(len(stream))]
    enc_orc = [enc[i] if i <= k else write_input(i, [w for w in stream[:i] if w.index != k])
               for i in range(len(stream))]
    decoy = (encode_pairs(tok, decoy_w.train_pairs() + replay_pairs(seed, k, args.replay)
                          + rehearsal_pairs(stream[:k], seed, k, args.rehearse), args.max_len)
             if decoy_w else None)

    tw = stream[k]
    tgt_direct, tgt_para = tw.scored(), tw.scored(para=True)
    general = [(q, "", a) for q, a in GENERAL[: args.n_general]]
    before_w = stream[:k]
    keep_before = [p for w in before_w for p in w.scored()]
    rng = random.Random(seed * 1000 + k)
    keep_before_gen = rng.sample(facts_items(before_w), min(args.n_gen_keep, len(keep_before)))

    def evaluate(S: Optional[State], m: int, gen: bool, relearn: bool) -> dict:
        after_w = stream[k + 1: k + m + 1]
        near_w = after_w[:4]
        keep_after = [p for w in after_w for p in w.scored()]
        if S is None:
            fast.reset()
        else:
            fast.load(S)
        r = {
            "nll_target_direct": answer_nll(model, tok, tgt_direct, args.max_len),
            "nll_target_para": answer_nll(model, tok, tgt_para, args.max_len),
            "nll_keep_before": answer_nll(model, tok, keep_before, args.max_len),
            "nll_keep_near": answer_nll(model, tok, [p for w in near_w for p in w.scored()], args.max_len),
            "nll_keep_after": answer_nll(model, tok, keep_after, args.max_len),
            "nll_general": answer_nll(model, tok, general, args.max_len),
        }
        if binary:
            r["p_true_target_direct"] = p_true(model, tok, tw.binary_items(), args.max_len)
            r["p_true_target_para"] = p_true(model, tok, tw.binary_items(para=True), args.max_len)
            r["p_true_keep_before"] = p_true(model, tok, [t for w in before_w for t in w.binary_items()],
                                             args.max_len)
            r["p_true_keep_after"] = p_true(model, tok, [t for w in after_w for t in w.binary_items()],
                                            args.max_len)
        if gen:
            seen_values = [f.value for w in stream[: k + m + 1] for f in w.facts]
            person_values = {w.entity: [f.value for f in w.facts] for w in stream}
            grng = random.Random(seed * 1000 + k * 100 + m)
            after_items = facts_items(after_w)
            keep_after_gen = grng.sample(after_items, min(args.n_gen_keep, len(after_items)))
            g = lambda items, texts: gen_eval(model, tok, binary, items, seen_values, person_values,  # noqa
                                              texts, args.max_new)
            r["gen_target_direct"] = g(facts_items([tw]), True)
            r["gen_target_para"] = g(facts_items([tw], para=True), True)
            r["gen_keep_before"] = g(keep_before_gen, False)
            r["gen_keep_after"] = g(keep_after_gen, False)
            r["gen_general"] = g([(q, key, "", "") for (q, _, _), key in zip(general, GENERAL_KEYS)], False)
        if relearn and args.relearn_steps > 0:
            rcfg = UpdateConfig(lr=args.lr, epochs=args.relearn_steps, batch_size=len(tgt_direct),
                                max_len=args.max_len, seed=seed)
            # relearn on the DIRECT phrasing only, test on the held-out paraphrase
            train(model, fast, encode_pairs(tok, tw.direct_train_pairs(), args.max_len), pad, rcfg)
            r["relearn_nll_target_para"] = answer_nll(model, tok, tgt_para, args.max_len)
            r["relearn_nll_target_direct"] = answer_nll(model, tok, tgt_direct, args.max_len)
            if binary:
                r["relearn_p_true_target_para"] = p_true(model, tok, tw.binary_items(para=True), args.max_len)
        return r

    # base (B = 0): generated answers once; NLL / p_true per m below, since keep_after grows with m
    base_gen = {n: v for n, v in evaluate(None, 0, gen=args.gen_eval, relearn=False).items() if n.startswith("gen_")}

    t0 = time.time()
    full_states, full_deltas, dB_decoy = full_pass(model, fast, enc, pad, cfg, k, last, checkpoints, decoy)
    t_full = time.time() - t0
    t0 = time.time()
    orc_states, orc_deltas = oracle_pass(model, fast, enc_orc, pad, cfg, k, last, checkpoints)
    t_orc = time.time() - t0
    rep_states, rep_deltas = None, None
    if args.repeat_full:
        rep_states, rep_deltas, _ = full_pass(model, fast, enc, pad, cfg, k, last, checkpoints, None)

    dB_k = full_deltas[k]
    dB_norm = max(state_norm(dB_k), 1e-12)
    R = random_like(dB_k, seed * 7919 + k) if args.diag else None

    if args.save_deltas:
        os.makedirs(args.save_deltas, exist_ok=True)
        path = os.path.join(args.save_deltas, f"{args.tag}_{args.kind}_seed{seed}_k{k}.pt")
        torch.save({"seed": seed, "k": k, "kind": args.kind, "checkpoints": checkpoints,
                    "full": full_deltas, "orc": orc_deltas, "decoy": dB_decoy, "random": R,
                    "repeat": rep_deltas}, path)

    for m in checkpoints:
        t0 = time.time()
        S_full, S_orc = full_states[m], orc_states[m]
        S_sub = state_axpy(S_full, dB_k, 1.0)
        diff_full_orc = {n: S_full[n] - S_orc[n] for n in S_full}
        alpha_star = state_dot(diff_full_orc, dB_k) / dB_norm ** 2
        resid = {n: S_sub[n] - S_orc[n] for n in S_sub}
        resid_norm = state_norm(resid)
        row = {
            "seed": seed, "k": k, "m": m, "kind": args.kind, "n_writes_seen": k + m + 1,
            "target_entity": tw.entity,
            "dB_k_norm": dB_norm,
            "dB_decoy_norm": state_norm(dB_decoy) if dB_decoy else None,
            "B_full_norm": state_norm(S_full),
            "residual_abs": resid_norm,
            "residual_rel": resid_norm / dB_norm,
            "full_vs_orc_rel": state_norm(diff_full_orc) / dB_norm,
            "alpha_star": alpha_star,
            "cos_resid_dB": state_dot(resid, dB_k) / max(resid_norm * dB_norm, 1e-12),
            "sec_full_pass": t_full, "sec_oracle_pass": t_orc,
        }
        row["base"] = {**evaluate(None, m, gen=False, relearn=False), **base_gen}
        if m == 0 and row["residual_rel"] > 1e-4:
            print(f"WARNING seed={seed} k={k}: m=0 residual_rel={row['residual_rel']:.2e} (expected 0); "
                  f"full and oracle prefixes differ", flush=True)
        if rep_states is not None:
            row["noise_rel"] = state_dist(S_full, rep_states[m]) / dB_norm
        for name, S in (("full", S_full), ("sub", S_sub), ("oracle", S_orc)):
            row[name] = evaluate(S, m, gen=args.gen_eval, relearn=name in ("sub", "oracle"))
        if args.diag:
            diag = {f"sub_a{a:g}": state_axpy(S_full, dB_k, a) for a in args.alphas}
            diag["sub_astar"] = state_axpy(S_full, dB_k, alpha_star)
            diag["sub_random"] = state_axpy(S_full, R, 1.0)
            diag["sub_decoy"] = state_axpy(S_full, dB_decoy, 1.0)
            if rep_states is not None:
                diag["sub_repeat"] = state_axpy(rep_states[m], rep_deltas[k], 1.0)
            for name, S in diag.items():
                row[name] = evaluate(S, m, gen=False, relearn=False)
        row["sec_eval"] = time.time() - t0
        fast.reset()
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        msg = (f"seed={seed} k={k} m={m:3d} resid_rel={row['residual_rel']:.3f} a*={alpha_star:.3f} "
               f"tgt_para nll full/sub/orc/base={row['full']['nll_target_para']:.3f}/"
               f"{row['sub']['nll_target_para']:.3f}/{row['oracle']['nll_target_para']:.3f}/"
               f"{row['base']['nll_target_para']:.3f} keep_after sub/orc={row['sub']['nll_keep_after']:.3f}/"
               f"{row['oracle']['nll_keep_after']:.3f} eval={row['sec_eval']:.0f}s")
        if binary:
            msg += (f" p_true_tgt full/sub/orc/base={row['full']['p_true_target_direct']:.2f}/"
                    f"{row['sub']['p_true_target_direct']:.2f}/{row['oracle']['p_true_target_direct']:.2f}/"
                    f"{row['base']['p_true_target_direct']:.2f}")
        print(msg, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=8, help="paper: 4")
    ap.add_argument("--kind", default="open", choices=KINDS)
    ap.add_argument("--n-writes", type=int, default=81)
    ap.add_argument("--facts-per-write", type=int, default=4)
    ap.add_argument("--targets", type=int, nargs="+", default=[16], help="indices k of the write to delete")
    ap.add_argument("--after", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lr", type=float, default=1e-3, help="paper: 5e-4")
    ap.add_argument("--epochs", type=int, default=10, help="paper: 5")
    ap.add_argument("--values", default="real", choices=["real", "made"])
    ap.add_argument("--answer-format", default="sentence", choices=["sentence", "word"], help="paper-like: word")
    ap.add_argument("--n-phrasings", type=int, default=4, help="training questions per fact (paper-like: 2)")
    ap.add_argument("--replay", type=int, default=8, help="general QA mixed into each write (paper: 0)")
    ap.add_argument("--rehearse", type=int, default=0, help="QA pairs from earlier writes retrained per write")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-general", type=int, default=20)
    ap.add_argument("--relearn-steps", type=int, default=2)
    ap.add_argument("--gen-eval", action="store_true", help="greedy answers + answer types (slower)")
    ap.add_argument("--n-gen-keep", type=int, default=16, help="generated keep questions per stratum")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--no-diag", dest="diag", action="store_false", help="skip alpha/random/decoy controls")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--repeat-full", action="store_true", help="second identical full pass (noise floor)")
    ap.add_argument("--save-deltas", default="", help="directory for per-write deltas (.pt)")
    ap.add_argument("--tag", default="del")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--out", default="runs/deletion.jsonl")
    args = ap.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    fast = attach_svd_lora(model, rank=args.rank, last_n_layers=args.last_n_layers)
    print(f"model={args.model} device={device} dtype={args.dtype} kind={args.kind} "
          f"fast_params={sum(p.numel() for p in fast.parameters())}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps({"config": vars(args)}) + "\n")
        for s in args.seeds:
            for k in args.targets:
                run_one(model, tok, fast, args, s, k, f)


if __name__ == "__main__":
    main()
