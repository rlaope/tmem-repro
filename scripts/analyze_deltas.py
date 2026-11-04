"""Direction analysis of saved per-write deltas (run_deletion.py --save-deltas). CPU only.

For a target k and later write j, D_j = dB_j(full) - dB_j(oracle) is how much write j was
trained differently because k was there. Exactly:
    B_sub(m) - B_orc(m) = sum_{j=k+1..k+m} D_j
    B_full(m) - B_orc(m) = dB_k + sum_j D_j

Printed / written per file:
  norm_by_write     ||dB_i(full)|| for every write (does learning shrink after the first writes?)
  cos_k_mean        cos(dB_k, mean of the other writes' deltas)   (shared "format" component)
  cos_decoy_k       cos(dB_decoy, dB_k)                            (how person-specific dB_k is)
  per m: resid_rel, cos(resid, dB_k), alpha_star, and per j: ||D_j||/||dB_k||, cos(D_j, dB_k),
         cos(D_j, mean)
  per module at the largest m: ||resid||/||dB_k||
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import torch

State = Dict[str, torch.Tensor]


def flat(s: State) -> torch.Tensor:
    return torch.cat([s[n].float().reshape(-1) for n in sorted(s)])


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


def analyze(d: dict) -> dict:
    k, full, orc = d["k"], d["full"], d["orc"]
    cps: List[int] = d["checkpoints"]
    dk = flat(full[k])
    others = [flat(full[i]) for i in sorted(full) if i != k]
    mean = torch.stack(others).mean(0) if others else torch.zeros_like(dk)
    out = {
        "seed": d["seed"], "k": k, "kind": d.get("kind"),
        "norm_by_write": [float(flat(full[i]).norm()) for i in sorted(full)],
        "dB_k_norm": float(dk.norm()),
        "cos_k_mean": cos(dk, mean),
        "cos_decoy_k": cos(flat(d["decoy"]), dk) if d.get("decoy") is not None else None,
        "cos_decoy_mean": cos(flat(d["decoy"]), mean) if d.get("decoy") is not None else None,
        "by_m": [],
    }
    if d.get("repeat"):
        out["repeat_max_rel"] = max(float((flat(full[i]) - flat(d["repeat"][i])).norm()) for i in full) / float(dk.norm())
    resid = torch.zeros_like(dk)
    j_rows = []
    for j in range(k + 1, k + max(cps) + 1):
        Dj = flat(full[j]) - flat(orc[j])
        resid = resid + Dj
        j_rows.append({"j": j, "D_rel": float(Dj.norm() / dk.norm()), "cos_D_dBk": cos(Dj, dk),
                       "cos_D_mean": cos(Dj, mean)})
        m = j - k
        if m in cps:
            out["by_m"].append({
                "m": m,
                "resid_rel": float(resid.norm() / dk.norm()),
                "cos_resid_dBk": cos(resid, dk),
                "alpha_star": float((dk + resid) @ dk / (dk @ dk)),
                "per_j": list(j_rows),
            })
    m_max = max(cps)
    if m_max > 0:
        per_mod = {}
        for n in sorted(full[k]):
            r = sum(full[j][n].float() - orc[j][n].float() for j in range(k + 1, k + m_max + 1))
            per_mod[n] = float(r.norm() / full[k][n].float().norm().clamp_min(1e-12))
        out["per_module_resid_rel_at_max_m"] = per_mod
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = []
    for f in args.files:
        r = analyze(torch.load(f, map_location="cpu", weights_only=False))
        rows.append(r)
        print(f"{f}: seed={r['seed']} k={r['k']} |dB_k|={r['dB_k_norm']:.4f} cos(dB_k,mean)={r['cos_k_mean']:.3f} "
              f"cos(decoy,dB_k)={r['cos_decoy_k']} repeat_max_rel={r.get('repeat_max_rel')}")
        nb = r["norm_by_write"]
        print("  |dB_i| first 4 / last 4:", [round(x, 4) for x in nb[:4]], [round(x, 4) for x in nb[-4:]])
        for b in r["by_m"]:
            top = sorted(b["per_j"], key=lambda x: -x["D_rel"])[:3]
            print(f"  m={b['m']:3d} resid_rel={b['resid_rel']:.3f} cos(resid,dB_k)={b['cos_resid_dBk']:+.3f} "
                  f"a*={b['alpha_star']:.3f} top D_j: " +
                  ", ".join(f"j={t['j']}:{t['D_rel']:.3f}(cos_k {t['cos_D_dBk']:+.2f})" for t in top))
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
