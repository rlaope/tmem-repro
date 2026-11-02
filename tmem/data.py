"""LongMemEval-S loader (cleaned release on Hugging Face)."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

LME_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)


@dataclass
class Episode:
    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    sessions: List[str]  # one rendered text block per haystack session


def render_session(turns: list, date: Optional[str]) -> str:
    head = f"[Session date: {date}]" if date else "[Session]"
    lines = [head]
    for t in turns:
        lines.append(f"{t.get('role', 'user')}: {t.get('content', '')}")
    return "\n".join(lines)


def download(path: str, url: str = LME_S_URL) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_longmemeval(path: str, limit: Optional[int] = None, skip_abstention: bool = False) -> List[Episode]:
    with open(path) as f:
        raw = json.load(f)
    eps: List[Episode] = []
    for r in raw:
        qid = str(r["question_id"])
        if skip_abstention and qid.endswith("_abs"):
            continue
        dates = r.get("haystack_dates") or [None] * len(r["haystack_sessions"])
        sessions = [render_session(s, d) for s, d in zip(r["haystack_sessions"], dates)]
        eps.append(
            Episode(
                qid=qid,
                qtype=r.get("question_type", ""),
                question=r["question"],
                answer=str(r["answer"]),
                question_date=r.get("question_date", ""),
                sessions=sessions,
            )
        )
        if limit and len(eps) >= limit:
            break
    return eps
