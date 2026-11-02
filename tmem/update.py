"""Online SFT operator T(Delta, QA) (TMEM Sec. 5.1 "TMEM implementation").

Paper setting: SGD, lr 5e-4, 5 epochs, batch size 16, only B is trained,
updates are cumulative within an episode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn

from .svd_lora import FastWeights

QAPair = Tuple[str, str]


@dataclass
class UpdateConfig:
    lr: float = 5e-4
    epochs: int = 5
    batch_size: int = 16
    max_len: int = 256
    seed: int = 0


def encode_qa(tokenizer, question: str, answer: str, max_len: int) -> Tuple[List[int], List[int]]:
    """Chat-format one QA pair; loss only on the answer tokens."""
    msgs = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token or ""
    a_ids = tokenizer(answer + eos, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


def _collate(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int, device):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, : len(i)] = torch.tensor(i)
        lab[k, : len(l)] = torch.tensor(l)
        att[k, : len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def online_sft_ids(
    model: nn.Module,
    fast: FastWeights,
    examples: Sequence[Tuple[List[int], List[int]]],
    pad_id: int,
    cfg: UpdateConfig,
) -> List[float]:
    """Run the update on pre-tokenized examples. Returns per-step losses."""
    if not examples:
        return []
    device = next(model.parameters()).device
    params = fast.parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=cfg.lr)
    rng = random.Random(cfg.seed)
    order = list(range(len(examples)))
    losses: List[float] = []
    was_training = model.training
    model.train(False)  # no dropout; gradients still flow to B
    for _ in range(cfg.epochs):
        rng.shuffle(order)
        for s in range(0, len(order), cfg.batch_size):
            batch = [examples[i] for i in order[s : s + cfg.batch_size]]
            ids, lab, att = _collate(batch, pad_id, device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.detach()))
    for p in params:
        p.requires_grad_(False)
    model.train(was_training)
    return losses


def online_sft(model, tokenizer, fast: FastWeights, qa: Sequence[QAPair], cfg: UpdateConfig) -> List[float]:
    examples = [encode_qa(tokenizer, q, a, cfg.max_len) for q, a in qa]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return online_sft_ids(model, fast, examples, pad, cfg)
