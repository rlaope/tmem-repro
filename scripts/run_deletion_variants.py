"""삭제 방식 비교: dB_k 전체를 빼는 대신, 모든 기억이 공유하는 성분(공통 변화)은 남기고 빼면 남은 기억 손상이 줄어드는가.

run_deletion.py --save-deltas 결과(.pt)에서 상태를 다시 만든다 (B는 0에서 시작하므로 delta의 합):
  B_full(m) = sum_{i<=k+m} dB_i(full)
  B_orc(m)  = sum_{i<k} dB_i(full) + sum_{k<j<=k+m} dB_j(oracle)

비교하는 삭제 방식 (B_full(m)에서 빼는 양):
  sub              dB_k                                   (기존: 전체 빼기)
  sub_minus_decoy  dB_k - dB_decoy                         (같은 시점에 학습한 무관한 사람의 변화량을 공통 성분으로 봄)
  sub_proj_decoy   dB_k - proj(dB_k on dB_decoy)           (dB_k 중 decoy 방향 성분만 남김)
  sub_proj_mean    dB_k - proj(dB_k on mean of other writes)  (decoy 없이, 다른 기억들 변화량의 평균 방향을 공통 성분으로 봄)
정답지는 oracle(k를 처음부터 안 배움). 평가는 생성 정답률 + 값 토큰 NLL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_deletion as rd  # noqa: E402
from tmem.svd_lora import attach_svd_lora  # noqa: E402
from tmem.synth import GENERAL, GENERAL_KEYS, make_stream_with_decoy  # noqa: E402

State = Dict[str, torch.Tensor]


def ssum(states):
    out = None
    for s in states:
        if out is None:
            out = {n: v.float().clone() for n, v in s.items()}
        else:
            for n in out:
                out[n] += s[n].float()
    return out


def flat(s: State) -> torch.Tensor:
    return torch.cat([s[n].reshape(-1) for n in sorted(s)])


def axpy(x: State, y: State, a: float) -> State:
    return {n: x[n] - a * y[n] for n in x}


def proj_coef(v: State, u: State) -> float:
    fu = flat(u)
    return float(flat(v) @ fu / (fu @ fu).clamp_min(1e-20))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-writes", type=int, default=81)
    ap.add_argument("--last-n-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--after", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--out", default="runs/del_variants.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    fast = attach_svd_lora(model, rank=args.rank, last_n_layers=args.last_n_layers)
    dev = next(model.parameters()).device
    general = [(q, "", a) for q, a in GENERAL]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "a")
    out_f.write(json.dumps({"config": vars(args)}) + "\n")

    for path in args.deltas:
        d = torch.load(path, map_location="cpu", weights_only=False)
        seed, k = d["seed"], d["k"]
        stream, _ = make_stream_with_decoy(args.n_writes, 4, seed, d.get("kind", "open"), values="real",
                                           answer_format="sentence", n_phrasings=4)
        full, orc, dB_decoy = d["full"], d["orc"], d["decoy"]
        dB_k = {n: v.float() for n, v in full[k].items()}
        tw = stream[k]
        before_w = stream[:k]
        person_values = {w.entity: [f.value for f in w.facts] for w in stream}

        def evaluate(S: State, m: int) -> dict:
            fast.load({n: v.to(dev) for n, v in S.items()})
            after_w = stream[k + 1: k + m + 1]
            seen = [f.value for w in stream[: k + m + 1] for f in w.facts]
            g = lambda items: rd.gen_eval(model, tok, False, items, seen, person_values, False, args.max_new)  # noqa
            r = {
                "acc_target_direct": g(rd.facts_items([tw]))["acc"],
                "acc_target_para": g(rd.facts_items([tw], para=True))["acc"],
                "acc_keep_before": g(rd.facts_items(before_w))["acc"],
                "acc_keep_after": g(rd.facts_items(after_w))["acc"],
                "acc_general": rd.gen_eval(model, tok, False, [(q, key, "", "") for (q, _, _), key in
                                                               zip(general, GENERAL_KEYS)], [], {}, False,
                                           args.max_new)["acc"],
                "nll_target_para": rd.answer_nll(model, tok, tw.scored(para=True), args.max_len),
                "nll_keep_before": rd.answer_nll(model, tok, [p for w in before_w for p in w.scored()], args.max_len),
                "nll_keep_after": rd.answer_nll(model, tok, [p for w in after_w for p in w.scored()], args.max_len),
            }
            fast.reset()
            return r

        for m in args.after:
            t0 = time.time()
            B_full = ssum([full[i] for i in range(k + m + 1)])
            B_orc = ssum([full[i] for i in range(k)] + [orc[j] for j in range(k + 1, k + m + 1)])
            others = [full[i] for i in range(k + m + 1) if i != k]
            mean_other = {n: v / len(others) for n, v in ssum(others).items()}
            c_dec, c_mean = proj_coef(dB_k, dB_decoy), proj_coef(dB_k, mean_other)
            removed = {
                "sub": dB_k,
                "sub_minus_decoy": axpy(dB_k, dB_decoy, 1.0),
                "sub_proj_decoy": axpy(dB_k, dB_decoy, c_dec),
                "sub_proj_mean": axpy(dB_k, mean_other, c_mean),
            }
            row = {"seed": seed, "k": k, "m": m, "proj_coef_decoy": c_dec, "proj_coef_mean": c_mean,
                   "full": evaluate(B_full, m), "oracle": evaluate(B_orc, m)}
            for name, R in removed.items():
                row[name] = evaluate(axpy(B_full, R, 1.0), m)
                row[name]["removed_norm_rel"] = float(flat(R).norm() / flat(dB_k).norm())
            row["sec"] = time.time() - t0
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            s = lambda v: f"{row[v]['acc_target_direct']:.2f}/{row[v]['acc_keep_before']:.2f}/{row[v]['acc_keep_after']:.2f}"  # noqa
            print(f"seed={seed} m={m} [지운기억/앞기억/뒤기억 정답률] full {s('full')} orc {s('oracle')} sub {s('sub')} "
                  f"-decoy {s('sub_minus_decoy')} projdec {s('sub_proj_decoy')} projmean {s('sub_proj_mean')} "
                  f"({row['sec']:.0f}s)", flush=True)
    out_f.close()


if __name__ == "__main__":
    main()
