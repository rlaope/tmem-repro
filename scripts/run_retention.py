"""Retention check: after N sequential writes, how much of each older write can the model still answer?

Deleting a memory only means something if the memory is still there. On Qwen3-0.6B (Mac probes) a write
is recalled right after training but is mostly gone 1-3 writes later. This script measures that curve
for a model and a list of write settings, so we can pick a model/setting before the deletion runs.

Per config (one JSON line):
  acc_now[i]          direct-question accuracy on write i right after training it
  acc_end_direct[i]   same question after all N writes
  acc_end_para[i]     held-out paraphrase after all N writes
  nll_end[i]          value-token NLL after all N writes
  by_age              mean acc_end_direct grouped by age (0 = last write, 1 = one before, ...)
  general_acc / general_nll before and after
Config string: answer_format,n_phrasings,replay,lr,epochs,last_n_layers[,rehearse]
  rehearse = number of QA pairs from earlier writes trained again with each new write (default 0).
  Mirrors an agent that keeps a QA log and replays part of it; sampled uniformly over earlier writes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_deletion as rd  # noqa: E402
from tmem.answers import classify_open  # noqa: E402
from tmem.svd_lora import attach_svd_lora  # noqa: E402
from tmem.synth import GENERAL, GENERAL_KEYS, make_stream  # noqa: E402
from tmem.update import UpdateConfig  # noqa: E402

DEFAULT_CONFIGS = ("word,2,0,5e-4,5,4;"       # paper-like
                   "sentence,4,8,1e-3,10,8;"   # current default
                   "sentence,4,8,5e-4,10,8;"
                   "sentence,4,8,2e-3,10,8")
AGE_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63)]


def rehearsal_pairs(earlier, seed: int, index: int, n: int):
    """n QA pairs drawn uniformly from earlier writes' training pairs; depends only on (seed, index)."""
    pool = [p for w in earlier for p in w.train_pairs()]
    if n <= 0 or not pool:
        return []
    return random.Random(seed * 7_000_003 + index).sample(pool, min(n, len(pool)))


def acc(model, tok, pairs, max_new):
    outs = rd.generate(model, tok, [q for q, _ in pairs], max_new=max_new)
    return [int(classify_open(o, a, []) == "gold") for (_, a), o in zip(pairs, outs)]


def general_acc(model, tok, max_new):
    outs = rd.generate(model, tok, [q for q, _ in GENERAL], max_new=max_new)
    return mean(int(k in o.lower()) for k, o in zip(GENERAL_KEYS, outs))


def run_config(args, tok, cfg_str: str, seed: int, out_f) -> None:
    parts = cfg_str.split(",")
    fmt, nq, rep, lr, ep, layers = parts[:6]
    rehearse = int(parts[6]) if len(parts) > 6 else 0
    nq, rep, lr, ep, layers = int(nq), int(rep), float(lr), int(ep), int(layers)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    fast = attach_svd_lora(model, rank=args.rank, last_n_layers=layers)
    stream = make_stream(args.n_writes, args.facts_per_write, seed, "open", values=args.values,
                         answer_format=fmt, n_phrasings=nq)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    general = [(q, "", a) for q, a in GENERAL]
    g_acc0, g_nll0 = general_acc(model, tok, args.max_new), rd.answer_nll(model, tok, general, args.max_len)
    ucfg = UpdateConfig(lr=lr, epochs=ep, batch_size=args.batch_size, max_len=args.max_len, seed=seed)
    acc_now = []
    for i, w in enumerate(stream):
        old = rehearsal_pairs(stream[:i], seed, i, rehearse)
        ex = rd.encode_pairs(tok, w.train_pairs() + rd.replay_pairs(seed, i, rep) + old, args.max_len)
        rd.train(model, fast, ex, pad, ucfg)
        acc_now.append(mean(acc(model, tok, w.direct_pairs(), args.max_new)))
    t_train = time.time() - t0
    acc_d = [mean(acc(model, tok, w.direct_pairs(), args.max_new)) for w in stream]
    acc_p = [mean(acc(model, tok, w.para_pairs(), args.max_new)) for w in stream]
    nll = [rd.answer_nll(model, tok, w.scored(), args.max_len) for w in stream]
    n = len(stream)
    by_age = {}
    for lo, hi in AGE_BUCKETS:
        idx = [n - 1 - a for a in range(lo, hi + 1) if 0 <= n - 1 - a]
        if idx:
            by_age[f"{lo}-{hi}"] = mean(acc_d[i] for i in idx)
    row = {
        "model": args.model, "seed": seed, "config": cfg_str, "rehearse": rehearse, "n_writes": n,
        "acc_now": acc_now, "acc_end_direct": acc_d, "acc_end_para": acc_p, "nll_end": nll, "by_age": by_age,
        "mean_acc_now": mean(acc_now),
        "general_acc_before": g_acc0, "general_acc_after": general_acc(model, tok, args.max_new),
        "general_nll_before": g_nll0, "general_nll_after": rd.answer_nll(model, tok, general, args.max_len),
        "sec_train": t_train, "sec_total": time.time() - t0,
    }
    out_f.write(json.dumps(row) + "\n")
    out_f.flush()
    print(f"{args.model} seed={seed} cfg={cfg_str} | right after write {row['mean_acc_now']:.2f} | after all "
          f"by age {{{', '.join(f'{k}: {v:.2f}' for k, v in by_age.items())}}} | general "
          f"{g_acc0:.2f}->{row['general_acc_after']:.2f} | {row['sec_total']:.0f}s", flush=True)
    del model, fast
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--n-writes", type=int, default=32)
    ap.add_argument("--facts-per-write", type=int, default=4)
    ap.add_argument("--values", default="real", choices=["real", "made"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="';'-separated fmt,phrasings,replay,lr,epochs,layers")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/retention.jsonl")
    args = ap.parse_args()
    args.device = rd.pick_device(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps({"config": vars(args)}) + "\n")
        for s in args.seeds:
            for c in args.configs.split(";"):
                if c.strip():
                    run_config(args, tok, c.strip(), s, f)


if __name__ == "__main__":
    main()
