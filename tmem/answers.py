"""Classify a generated answer after a deletion.

Open facts (made-up answer words), checked in this order:
  gold          the asked fact's own value appears (for a deleted fact: it was not erased);
                spaces/hyphens are ignored, so "Kel-yorr" counts
  partial       a word in the answer shares a 4+ letter prefix with the value (partial recall)
  self_other    another fact of the same person (e.g. their city when asked for their pet)
  other_memory  a value of another person in the stream (borrowed memory)
  refuse        says it does not know / has no information
  garbled       empty or mostly non-text / one word repeated
  other         anything else (a made-up answer the stream never taught)

Binary facts (two options): true / wrong / both / refuse / garbled / other.
Each option also matches short aliases (오른손잡이 ~ 오른손, 남자 ~ 남성), and an echoed
question is removed first, so "아침형인가요, 저녁형인가요? 저녁형입니다" is not "both".
The rules are string heuristics; generated target texts are saved so they can be checked by hand.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

OPEN_TYPES = ("gold", "partial", "self_other", "other_memory", "refuse", "garbled", "other")
BINARY_TYPES = ("true", "wrong", "both", "refuse", "garbled", "other")

BINARY_ALIASES = {
    "남자": ("남자", "남성"),
    "여자": ("여자", "여성"),
    "왼손잡이": ("왼손잡이", "왼손"),
    "오른손잡이": ("오른손잡이", "오른손"),
    "아침형": ("아침형", "아침"),
    "저녁형": ("저녁형", "저녁"),
    "강아지": ("강아지",),
    "고양이": ("고양이",),
}

_REFUSE = re.compile(
    r"(don't know|do not know|not sure|no information|don't have|do not have|cannot|can't|unknown|"
    r"not available|unable|sorry|not provided|not specified|not mentioned|no record|"
    r"모르|알 수 없|정보가 없|정보는 없|죄송|확인할 수 없|제공되지|알려지지)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[0-9A-Za-z가-힣]")
_SEP = re.compile(r"[\s\-_'’.*]+")


def is_refusal(text: str) -> bool:
    return bool(_REFUSE.search(text))


def is_garbled(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(_WORD.findall(t)) < 0.5 * len(t.replace(" ", "")):
        return True
    words = t.split()
    return len(words) >= 3 and max(words.count(w) for w in set(words)) >= 0.6 * len(words)


def _norm(s: str) -> str:
    return _SEP.sub("", s.lower())


def _contains(text: str, value: str) -> bool:
    return value.lower() in text.lower() or _norm(value) in _norm(text)


def _partial(text: str, value: str, min_prefix: int = 4) -> bool:
    v = _norm(value)
    if len(v) < min_prefix + 1:
        return False
    for w in re.findall(r"[a-z]+", text.lower()):
        n = 0
        while n < min(len(w), len(v)) and w[n] == v[n]:
            n += 1
        if n >= min_prefix:
            return True
    return False


def classify_open(text: str, gold: str, other_values: Iterable[str], self_values: Sequence[str] = ()) -> str:
    if _contains(text, gold):
        return "gold"
    if _partial(text, gold):
        return "partial"
    g = gold.lower()
    if any(_contains(text, v) for v in self_values if v.lower() != g):
        return "self_other"
    if any(_contains(text, v) for v in other_values if v.lower() != g):
        return "other_memory"
    if is_refusal(text):
        return "refuse"
    if is_garbled(text):
        return "garbled"
    return "other"


def _strip_question(text: str, question: str) -> str:
    if not question:
        return text
    t = text.replace(question, " ")
    q = question.rstrip("?？ ")
    return t.replace(q, " ") if q else t


def classify_binary(text: str, true: str, wrong: str, question: str = "") -> str:
    t = _strip_question(text, question)
    has_t = any(a in t for a in BINARY_ALIASES.get(true, (true,)))
    has_w = any(a in t for a in BINARY_ALIASES.get(wrong, (wrong,)))
    if has_t and has_w:
        return "refuse" if is_refusal(t) else "both"
    if has_t:
        return "true"
    if has_w:
        return "wrong"
    if is_refusal(t):
        return "refuse"
    if is_garbled(t):
        return "garbled"
    return "other"
