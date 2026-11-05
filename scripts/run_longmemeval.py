"""Evaluate a memory strategy on LongMemEval-S.

Example (paper setting, one GPU):
    python scripts/run_longmemeval.py --model Qwen/Qwen3-4B --strategy tmem --l-max 12288 \
        --out runs/lme_tmem_4b.jsonl

Resumable: episodes already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tmem.agent import AgentConfig, MemoryAgent, result_to_json  # noqa: E402
from tmem.data import download, load_longmemeval  # noqa: E402
from tmem.metrics import exact_match, f1  # noqa: E402
from tmem.svd_lora import attach_svd_lora  # noqa: E402
from tmem.update import UpdateConfig  # noqa: E402


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--strategy", choices=["tmem", "summary", "none"], default="tmem")
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-sessions", type=int, default=None, help="smoke tests only: keep the last N haystack sessions")
    ap.add_argument("--l-max", type=int, default=12288)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-qa-history", action="store_true")
    ap.add_argument("--answer-qa", choices=["none", "all"], default="none",
                    help="also show extracted QA pairs as text when answering")
    ap.add_argument("--no-delta-at-answer", action="store_true", help="reset Delta before answering")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
    model.eval()

    fast = attach_svd_lora(model, rank=args.rank, last_n_layers=args.last_n_layers) if args.strategy == "tmem" else None
    cfg = AgentConfig(
        strategy=args.strategy,
        l_max=args.l_max,
        max_context=args.max_context,
        qa_history_in_prompt=not args.no_qa_history,
        answer_qa=args.answer_qa,
        delta_at_answer=not args.no_delta_at_answer,
        update=UpdateConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed),
    )
    agent = MemoryAgent(model, tok, fast, cfg)

    episodes = load_longmemeval(download(args.data), limit=args.limit)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "a") as out:
        for i, ep in enumerate(episodes):
            if ep.qid in done:
                continue
            sessions = ep.sessions[-args.max_sessions :] if args.max_sessions else ep.sessions
            res = agent.run_episode(ep.question, sessions, ep.question_date)
            scores = {"f1": f1(res.prediction, ep.answer), "em": exact_match(res.prediction, ep.answer)}
            out.write(result_to_json(ep.qid, ep.answer, res, scores) + "\n")
            out.flush()
            print(
                f"[{i + 1}/{len(episodes)}] {ep.qid} f1={scores['f1']:.2f} writes={len(res.writes)} "
                f"{res.seconds:.1f}s pred={res.prediction[:60]!r}",
                flush=True,
            )

    rows = [json.loads(l) for l in open(args.out) if l.strip()]
    n = len(rows)
    print(json.dumps({"n": n, "f1": 100 * sum(r["f1"] for r in rows) / n, "em": 100 * sum(r["em"] for r in rows) / n}))


if __name__ == "__main__":
    main()
