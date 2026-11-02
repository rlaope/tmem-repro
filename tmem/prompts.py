"""Prompts. EXTRACT_* follow TMEM Figure 1 and Appendix D (Figure 6)."""
from __future__ import annotations

import json
import re
from typing import List, Tuple

EXTRACT_SYSTEM = (
    "You are given one problem to solve, previous extracted QA pairs and one conversation session. "
    "Your task is to create high-quality supervised fine-tuning (SFT) QA pairs grounded ONLY in this session.\n\n"
    "Question:\n<question>\n{prompt}\n</question>\n\n"
    "Previous extracted QA pairs:\n<qa_history>\n{qa_history}\n</qa_history>\n\n"
    "Session:\n<session>\n{chunk}\n</session>"
)

EXTRACT_INSTRUCTION = """Task: Generate grounded SFT QA pairs from the current session.
Given the problem to solve, previous conversation history. Now you should create high-quality supervised fine-tuning (SFT) QA pairs grounded on the history.
Requirements:
1. Generate QA pairs adaptively based on how much useful information is present in the session.
 - If the session contains rich, concrete facts, generate more QA pairs.
 - If the session has limited useful evidence, generate fewer QA pairs.
 - If there is no usable evidence, return an empty JSON array.
2. You can generate QA pairs that capture the lessons learned from the session to help improve future interactions, such as preferences, plans, events, and temporal details, rather than just factual questions.
3. Each question must be answerable using explicit information from the session.
4. Each answer must be concise, factual, and directly supported by the session.
5. Cover diverse types when possible: who/what/when/where, preferences, plans, events, and temporal details.
6. Avoid duplicate or near-duplicate QA pairs, and keep wording natural and clear.
Return ONLY a JSON array. Each item must be:
{ "instruction": "<question>", "output": "<answer>" }
Output the generated SFT QA pairs in the specified JSON format. Do not include any explanations or additional text."""

# Summary baseline (MemAgent-style overwrite). Not specified verbatim in the paper.
SUMMARY_SYSTEM = (
    "You are reading a long conversation chunk by chunk to answer a question later.\n\n"
    "Question:\n<question>\n{prompt}\n</question>\n\n"
    "Current memory:\n<memory>\n{memory}\n</memory>\n\n"
    "New chunk:\n<session>\n{chunk}\n</session>"
)
SUMMARY_INSTRUCTION = (
    "Rewrite the memory so it keeps every fact from the old memory and the new chunk that could help "
    "answer the question, including dates and numbers. Output only the updated memory."
)

ANSWER_SYSTEM = (
    "Answer the question about the user's past conversations. "
    "Give a short answer with only the essential words.\n\n"
    "{memory_block}"
    "Recent conversation:\n<history>\n{history}\n</history>"
)


def format_qa_history(qa: List[Tuple[str, str]], limit: int = 50) -> str:
    items = [{"instruction": q, "output": a} for q, a in qa[-limit:]]
    return json.dumps(items, ensure_ascii=False)


_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_qa_json(text: str) -> List[Tuple[str, str]]:
    """Parse the model's JSON array. Tolerates code fences and trailing prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = _ARRAY.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # fall back to object-by-object
        data = []
        for obj in re.findall(r"\{[^{}]*\}", m.group(0)):
            try:
                data.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
    out: List[Tuple[str, str]] = []
    seen = set()
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        q = str(item.get("instruction", "")).strip()
        a = str(item.get("output", "")).strip()
        if q and a and (q.lower(), a.lower()) not in seen:
            seen.add((q.lower(), a.lower()))
            out.append((q, a))
    return out
