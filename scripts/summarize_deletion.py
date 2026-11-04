"""Aggregate run_deletion.py JSONL over seeds: mean +- std per (kind, k, m).

Columns (sub minus oracle unless noted; positive keep gap = subtraction hurt kept memories):
  resid, a*          residual_rel, alpha_star
  tgt                nll_target_para: sub - oracle (negative = deleted fact still more likely than oracle)
  keepB / keepA      nll_keep_before / nll_keep_after gap
  rand / decoy       same keepA gap for sub_random / sub_decoy (controls)
  noise              keepA gap of sub_repeat vs sub (identical procedure twice)
  acc / types        --gen-eval: target direct accuracy for full/sub/oracle and sub's answer types
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from statistics import mean, pstdev


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "   n/a      "
    return f"{mean(xs):+.3f}±{pstdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:+.3f}      "


def gap(r, a, b, key):
    if a not in r or b not in r:
        return None
    return r[a][key] - r[b][key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    groups = defaultdict(list)
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            if "m" in r:
                groups[(r.get("kind", "open"), r["k"], r["m"])].append(r)
    hdr = ["kind", "k", "m", "n", "resid", "a*", "tgt", "keepB", "keepA", "rand", "decoy", "noise"]
    print(" ".join(f"{h:>13}" if i > 3 else f"{h:>9}" for i, h in enumerate(hdr)))
    for (kind, k, m), rs in sorted(groups.items()):
        cells = [
            ms([r["residual_rel"] for r in rs]), ms([r["alpha_star"] for r in rs]),
            ms([gap(r, "sub", "oracle", "nll_target_para") for r in rs]),
            ms([gap(r, "sub", "oracle", "nll_keep_before") for r in rs]),
            ms([gap(r, "sub", "oracle", "nll_keep_after") for r in rs]),
            ms([gap(r, "sub_random", "oracle", "nll_keep_after") for r in rs]),
            ms([gap(r, "sub_decoy", "oracle", "nll_keep_after") for r in rs]),
            ms([gap(r, "sub_repeat", "sub", "nll_keep_after") for r in rs]),
        ]
        print(f"{kind:>9} {k:>9} {m:>9} {len(rs):>9} " + " ".join(f"{c:>13}" for c in cells))
        if "gen_target_direct" in rs[0]["sub"]:
            acc = {v: mean(r[v]["gen_target_direct"]["acc"] for r in rs) for v in ("full", "sub", "oracle")}
            types = Counter()
            for r in rs:
                types.update(r["sub"]["gen_target_direct"]["types"])
            print(f"{'':>39} acc tgt full/sub/orc={acc['full']:.2f}/{acc['sub']:.2f}/{acc['oracle']:.2f} "
                  f"sub types={dict(types)}")
        if kind == "binary_ko":
            p = {v: mean(r[v]["p_true_target_direct"] for r in rs) for v in ("full", "sub", "oracle")}
            pb = mean(r["base"]["p_true_target_direct"] for r in rs)
            print(f"{'':>39} P(true) tgt base/full/sub/orc={pb:.2f}/{p['full']:.2f}/{p['sub']:.2f}/{p['oracle']:.2f}")


if __name__ == "__main__":
    main()
