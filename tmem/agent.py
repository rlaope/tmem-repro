"""Fast-weight rollout for conversational memory (TMEM Sec. 3.1, Eq. 2-6).

Strategies
- tmem:    at each trigger, extract QA pairs -> online SFT into Delta_t, clear h_t, carry no text.
- summary: at each trigger, rewrite a text memory m_t (MemAgent-style), Delta_t stays 0.
- none:    no memory; answer from the most recent `max_context` tokens only.

A trigger fires when len(h_t) + len(m_t) > L_max (in tokens).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from . import prompts as P
from .svd_lora import FastWeights
from .update import UpdateConfig, online_sft


@dataclass
class AgentConfig:
    strategy: str = "tmem"  # tmem | summary | none
    l_max: int = 12288
    max_context: int = 32768  # only used by `none`
    extract_max_new_tokens: int = 1024
    summary_max_new_tokens: int = 1024
    answer_max_new_tokens: int = 64
    qa_history_in_prompt: bool = True  # Appendix D prompt shows previous QA pairs during extraction
    # The paper does not show its answer prompt; Fig. 8 mentions "retrieved memory QA pairs".
    # answer_qa="all" also puts every extracted QA pair into the answer prompt as text.
    answer_qa: str = "none"  # none | all
    delta_at_answer: bool = True  # False: reset Delta before answering (text-only memory)
    update: UpdateConfig = field(default_factory=UpdateConfig)


@dataclass
class WriteRecord:
    """One memory write. Kept so a later write can be replayed or reverted (deletion experiments)."""

    index: int
    session_ids: List[int]
    qa: List[Tuple[str, str]]
    losses: List[float]
    delta_norm_after: float
    seconds: float
    dropped_dup: int = 0


@dataclass
class EpisodeResult:
    prediction: str
    writes: List[WriteRecord]
    seconds: float
    peak_mem_gb: Optional[float]


class MemoryAgent:
    def __init__(self, model, tokenizer, fast: Optional[FastWeights], cfg: AgentConfig):
        if cfg.strategy == "tmem" and fast is None:
            raise ValueError("tmem strategy needs fast weights (attach_svd_lora)")
        self.model = model
        self.tok = tokenizer
        self.fast = fast
        self.cfg = cfg
        self.last_extract: dict = {}

    # ---------- helpers ----------
    def ntok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate(self, system: str, user: str, max_new_tokens: int) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return text.strip()

    def split_long(self, text: str) -> List[str]:
        """Split a block that alone exceeds L_max into token windows."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.cfg.l_max:
            return [text]
        step = max(1, self.cfg.l_max - 64)
        return [self.tok.decode(ids[i : i + step]) for i in range(0, len(ids), step)]

    # ---------- memory operations ----------
    def write_tmem(self, question: str, chunk: str, qa_log: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[float]]:
        assert self.fast is not None
        system = P.EXTRACT_SYSTEM.format(
            prompt=question,
            qa_history=P.format_qa_history(qa_log) if self.cfg.qa_history_in_prompt else "[]",
            chunk=chunk,
        )
        raw = self.generate(system, P.EXTRACT_INSTRUCTION, self.cfg.extract_max_new_tokens)
        parsed = P.parse_qa_json(raw)
        # Small models tend to copy <qa_history> back verbatim; do not re-train on pairs already written.
        seen = {(q.lower(), a.lower()) for q, a in qa_log}
        qa = [(q, a) for q, a in parsed if (q.lower(), a.lower()) not in seen]
        self.last_extract = {"raw": raw, "parsed": len(parsed), "dropped_dup": len(parsed) - len(qa)}
        losses = online_sft(self.model, self.tok, self.fast, qa, self.cfg.update)
        return qa, losses

    def write_summary(self, question: str, chunk: str, memory: str) -> str:
        system = P.SUMMARY_SYSTEM.format(prompt=question, memory=memory or "(empty)", chunk=chunk)
        return self.generate(system, P.SUMMARY_INSTRUCTION, self.cfg.summary_max_new_tokens)

    # ---------- episode ----------
    def run_episode(self, question: str, sessions: List[str], question_date: str = "") -> EpisodeResult:
        cfg = self.cfg
        if self.fast is not None:
            self.fast.reset()  # TMEM: Delta starts at 0 every episode
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        h: List[str] = []
        h_ids: List[int] = []
        memory = ""
        qa_log: List[Tuple[str, str]] = []
        writes: List[WriteRecord] = []

        def h_len() -> int:
            return sum(self.ntok(x) for x in h)

        def trigger() -> None:
            nonlocal h, h_ids, memory
            if not h:
                return
            chunk = "\n\n".join(h)
            ts = time.time()
            if cfg.strategy == "tmem":
                assert self.fast is not None
                qa, losses = self.write_tmem(question, chunk, qa_log)
                qa_log.extend(qa)
                writes.append(
                    WriteRecord(
                        len(writes), list(h_ids), qa, losses, self.fast.norm(), time.time() - ts,
                        dropped_dup=self.last_extract["dropped_dup"],
                    )
                )
            elif cfg.strategy == "summary":
                memory = self.write_summary(question, chunk, memory)
                writes.append(WriteRecord(len(writes), list(h_ids), [], [], 0.0, time.time() - ts))
            h, h_ids = [], []

        for sid, sess in enumerate(sessions):
            for piece in self.split_long(sess):
                if cfg.strategy == "none":
                    h.append(piece)
                    h_ids.append(sid)
                    continue
                if h and h_len() + self.ntok(piece) + self.ntok(memory) > cfg.l_max:
                    trigger()
                h.append(piece)
                h_ids.append(sid)

        history = "\n\n".join(h)
        if cfg.strategy == "none":
            ids = self.tok(history, add_special_tokens=False)["input_ids"]
            history = self.tok.decode(ids[-cfg.max_context :])
        memory_block = f"Memory:\n<memory>\n{memory}\n</memory>\n\n" if memory else ""
        if cfg.strategy == "tmem" and cfg.answer_qa == "all" and qa_log:
            lines = "\n".join(f"- Q: {q} A: {a}" for q, a in qa_log)
            memory_block = f"Memory QA pairs:\n<memory>\n{lines}\n</memory>\n\n"
        if self.fast is not None and not cfg.delta_at_answer:
            self.fast.reset()
        system = P.ANSWER_SYSTEM.format(memory_block=memory_block, history=history)
        user = f"Current date: {question_date}\nQuestion: {question}" if question_date else f"Question: {question}"
        pred = self.generate(system, user, cfg.answer_max_new_tokens)

        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        return EpisodeResult(pred, writes, time.time() - t0, peak)


def result_to_json(qid: str, gold: str, res: EpisodeResult, scores: Dict[str, float]) -> str:
    return json.dumps(
        {
            "qid": qid,
            "gold": gold,
            "pred": res.prediction,
            **scores,
            "seconds": round(res.seconds, 2),
            "peak_mem_gb": res.peak_mem_gb,
            "n_writes": len(res.writes),
            "n_qa": sum(len(w.qa) for w in res.writes),
            "writes": [w.__dict__ for w in res.writes],
        },
        ensure_ascii=False,
    )
