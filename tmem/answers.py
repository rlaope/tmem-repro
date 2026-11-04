"""Classify a generated answer after a deletion.

Open facts (made-up answer words):
  gold          the asked fact's own value appears (for a deleted fact: it was not erased)
  other_memory  no gold, but another stream value appears (borrowed from another person)
  refuse        says it does not know / has no information
  garbled       empty or mostly non-text / one word repeated
  other         anything else (a made-up answer the stream never taught)

Binary facts (two options): true / wrong / both / refuse / garbled / other.
The rules are string heuristics; generated texts are saved so they can be checked by hand.
"""
from __future__ import annotations

import re
from typing import Iterable

OPEN_TYPES = ("gold", "other_memory", "refuse", "garbled", "other")
BINARY_TYPES = ("true", "wrong", "both", "refuse", "garbled", "other")

_REFUSE = re.compile(
    r"(don't know|do not know|not sure|no information|don't have|do not have|cannot|can't|unknown|"
    r"not available|unable|sorry|not provided|not specified|not mentioned|no record|"
    r"모르|알 수 없|정보가 없|정보는 없|죄송|확인할 수 없|제공되지|알려지지)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[0-9A-Za-z가-힣]")


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


def classify_open(text: str, gold: str, other_values: Iterable[str]) -> str:
    low = text.lower()
    if gold.lower() in low:
        return "gold"
    if any(v.lower() in low for v in other_values if v.lower() != gold.lower()):
        return "other_memory"
    if is_refusal(text):
        return "refuse"
    if is_garbled(text):
        return "garbled"
    return "other"


def classify_binary(text: str, true: str, wrong: str) -> str:
    has_t, has_w = true in text, wrong in text
    if has_t and has_w:
        return "both"
    if has_t:
        return "true"
    if has_w:
        return "wrong"
    if is_refusal(text):
        return "refuse"
    if is_garbled(text):
        return "garbled"
    return "other"
