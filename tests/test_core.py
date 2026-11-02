import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tmem.metrics import exact_match, f1
from tmem.prompts import parse_qa_json
from tmem.svd_lora import SVDLoRALinear, attach_svd_lora
from tmem.update import UpdateConfig, online_sft_ids


def tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).eval()


def test_svd_init_matches_top_singular_directions():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 48, bias=False)
    m = SVDLoRALinear(base, rank=4)
    _, S, Vh = torch.linalg.svd(base.weight.detach(), full_matrices=False)
    assert torch.allclose(m.A, S[:4, None] * Vh[:4], atol=1e-5)
    assert torch.count_nonzero(m.B) == 0


def test_attach_starts_at_base_policy():
    model = tiny_model()
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    fast = attach_svd_lora(model, rank=6, last_n_layers=4)
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-5)
    assert len(fast.modules) == 4 * 3
    assert all(".mlp." in k for k in fast.modules)
    assert {int(k.split(".")[1]) for k in fast.modules} == {2, 3, 4, 5}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == [], trainable


def _examples():
    torch.manual_seed(1)
    ex = []
    for _ in range(8):
        q = torch.randint(0, 128, (6,)).tolist()
        a = torch.randint(0, 128, (3,)).tolist()
        ex.append((q + a, [-100] * 6 + a))
    return ex


def test_online_sft_only_moves_B_and_reduces_loss():
    model = tiny_model()
    fast = attach_svd_lora(model, rank=6, last_n_layers=4)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters()}
    A_before = {k: m.A.clone() for k, m in fast.modules.items()}
    losses = online_sft_ids(model, fast, _examples(), pad_id=0, cfg=UpdateConfig(lr=0.5, epochs=20, batch_size=4))
    assert losses[-1] < losses[0]
    assert fast.norm() > 0
    for n, p in model.named_parameters():
        if n.endswith(".B"):
            continue
        assert torch.equal(p, frozen[n]), n
    for k, m in fast.modules.items():
        assert torch.equal(m.A, A_before[k])


def test_state_roundtrip_and_reset():
    model = tiny_model()
    fast = attach_svd_lora(model, rank=6, last_n_layers=4)
    online_sft_ids(model, fast, _examples(), pad_id=0, cfg=UpdateConfig(lr=0.5, epochs=2, batch_size=4))
    snap = fast.state()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    fast.reset()
    assert fast.norm() == 0
    fast.load(snap)
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, ref, atol=1e-6)


def test_update_is_deterministic_given_seed():
    def run():
        model = tiny_model()
        fast = attach_svd_lora(model, rank=6, last_n_layers=4)
        online_sft_ids(model, fast, _examples(), pad_id=0, cfg=UpdateConfig(lr=0.5, epochs=3, batch_size=4, seed=7))
        return fast.state()

    a, b = run(), run()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_parse_qa_json_variants():
    raw = '```json\n[{"instruction": "Q1?", "output": "A1"}, {"instruction": "Q1?", "output": "A1"}, {"x": 1}]\n```'
    assert parse_qa_json(raw) == [("Q1?", "A1")]
    assert parse_qa_json("<think>hm</think>[]") == []
    assert parse_qa_json("no json here") == []
    broken = '[{"instruction": "Q", "output": "A"}, {"instruction": "R", "output": ]'
    assert parse_qa_json(broken) == [("Q", "A")]


def test_metrics():
    assert exact_match("The cat.", "the cat") == 1.0
    assert f1("4", "4 years") == pytest.approx(2 / 3)
    assert f1("", "x") == 0.0
