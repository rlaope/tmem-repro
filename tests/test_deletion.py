"""Identities the deletion experiment relies on, checked on a tiny random Qwen3."""
import importlib.util
import os

import torch

from tmem.svd_lora import attach_svd_lora, state_dist, state_norm
from tmem.update import UpdateConfig

from test_core import tiny_model

_spec = importlib.util.spec_from_file_location(
    "run_deletion", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_deletion.py"))
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def _stream(n_writes=7, per_write=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_writes):
        ex = []
        for _ in range(per_write):
            q = torch.randint(0, 128, (6,), generator=g).tolist()
            a = torch.randint(0, 128, (3,), generator=g).tolist()
            ex.append((q + a, [-100] * 6 + a))
        out.append(ex)
    return out


def _setup():
    model = tiny_model()
    fast = attach_svd_lora(model, rank=4, last_n_layers=2)
    cfg = UpdateConfig(lr=5e-2, epochs=2, batch_size=4, seed=0)
    return model, fast, cfg


def test_passes_match_at_m0_and_residual_decomposes():
    model, fast, cfg = _setup()
    enc, k, cps = _stream(), 2, [0, 1, 2, 4]
    last = k + max(cps)
    full, fd, _ = rd.full_pass(model, fast, enc, 0, cfg, k, last, cps)
    orc, od = rd.oracle_pass(model, fast, enc, 0, cfg, k, last, cps)
    dB = fd[k]
    assert state_norm(dB) > 0
    # m = 0: subtracting equals the oracle exactly
    assert state_dist(rd.state_axpy(full[0], dB, 1.0), orc[0]) < 1e-6
    # m > 0: B_sub - B_orc == sum_j (dB_j^full - dB_j^orc), j in k+1..k+m
    for m in (1, 2, 4):
        resid = {n: rd.state_axpy(full[m], dB, 1.0)[n] - orc[m][n] for n in dB}
        acc = {n: torch.zeros_like(v) for n, v in dB.items()}
        for j in range(k + 1, k + m + 1):
            for n in acc:
                acc[n] += fd[j][n] - od[j][n]
        assert state_dist(resid, acc) < 1e-5
    # later writes did change once k was removed (otherwise the experiment is trivial)
    assert state_dist(rd.state_axpy(full[4], dB, 1.0), orc[4]) > 1e-6


def test_decoy_leaves_main_stream_unchanged():
    model, fast, cfg = _setup()
    enc, k, cps = _stream(), 2, [0, 2]
    decoy = _stream(1, seed=9)[0]
    a, _, dd = rd.full_pass(model, fast, enc, 0, cfg, k, k + 2, cps, decoy)
    b, _, none = rd.full_pass(model, fast, enc, 0, cfg, k, k + 2, cps, None)
    assert none is None and state_norm(dd) > 0
    for m in cps:
        assert state_dist(a[m], b[m]) < 1e-7


def test_full_pass_is_repeatable():
    model, fast, cfg = _setup()
    enc, k, cps = _stream(), 1, [0, 3]
    a, _, _ = rd.full_pass(model, fast, enc, 0, cfg, k, k + 3, cps)
    b, _, _ = rd.full_pass(model, fast, enc, 0, cfg, k, k + 3, cps)
    assert state_dist(a[3], b[3]) == 0.0


def test_random_like_matches_module_norms_and_alpha_star():
    ref = {"a": torch.randn(5, 3), "b": torch.randn(7, 3) * 3}
    r = rd.random_like(ref, seed=0)
    for n in ref:
        assert abs(float(r[n].norm()) - float(ref[n].norm())) < 1e-4
    # alpha_star recovers a known scale
    x = {n: 0.4 * v for n, v in ref.items()}
    assert abs(rd.state_dot(x, ref) / state_norm(ref) ** 2 - 0.4) < 1e-6
