"""Synthetic fact streams for deletion experiments.

One write = one fictional person with F attributes (like one session about one person).
Each fact has two training phrasings and one held-out paraphrase used only for evaluation.

Two kinds of stream:
  * "open":      English questions, made-up answer words. Every (attribute, value) is used
                 by exactly one person, so the base model cannot guess an answer and no
                 other person in the stream shares it.
  * "binary_ko": Korean questions about fictional Korean names, each fact has exactly two
                 possible answers (e.g. 남자/여자). Values are balanced 50/50 per attribute.
                 Measured by the probability of the true option, not by the generated text,
                 because the base model can guess right half the time.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- open (English)

ATTRS = {
    # attr: ([two training questions], held-out paraphrase, capitalize value)
    "pet": (["What is the name of {e}'s pet?", "{e} has a pet. What is it called?"],
            "What do they call the animal that lives with {e}?", True),
    "city": (["Which city does {e} live in?", "Where does {e} live?"],
             "{e} is a resident of which city?", True),
    "job": (["What does {e} do for a living?", "What is {e}'s job?"],
            "What is {e}'s profession?", False),
    "fruit": (["What is {e}'s favorite fruit?", "Which fruit does {e} like most?"],
              "If {e} could only eat one fruit, which would it be?", False),
    "instrument": (["Which instrument does {e} play?", "What instrument is {e} learning?"],
                   "{e} practices which musical instrument?", False),
    "car": (["What car does {e} drive?", "Which car brand does {e} own?"],
            "{e} gets around in a car made by which company?", True),
}

# Person names and answer values use disjoint syllable sets, and every value is checked
# not to occur inside any name or other value, so substring matching of answers is unambiguous.
SYL = ["ka", "zo", "mir", "tel", "va", "dru", "nos", "pel", "qui", "ran", "sev", "tor", "ul", "wex", "yan", "bri",
       "cal", "fen", "gor", "hal", "jas", "lum", "ost", "rik"]
VSYL = ["bex", "dov", "eth", "gim", "hob", "ix", "jun", "kel", "lom", "mab", "nev", "obb", "pim", "rux", "sab",
        "tiv", "ub", "vog", "wim", "yorr", "zed", "aph", "cus", "dil"]

# ---------------------------------------------------------------- binary (Korean)


def _has_batchim(word: str) -> bool:
    code = ord(word[-1]) - 0xAC00
    return 0 <= code < 11172 and code % 28 != 0


def _josa(name: str) -> Dict[str, str]:
    b = _has_batchim(name)
    return {"e": name, "e_eun": name + ("은" if b else "는"), "e_i": name + ("이" if b else "가"), "e_ui": name + "의"}


BINARY_KO = {
    # attr: ([two training questions], held-out paraphrase, (option_a, option_b))
    "gender": (["{e_ui} 성별은 무엇인가요?", "{e_eun} 남자인가요, 여자인가요?"],
               "{e_ui} 성별을 남자와 여자 중에서 고르면?", ("남자", "여자")),
    "hand": (["{e_eun} 어느 손을 주로 쓰나요?", "{e_eun} 왼손잡이인가요, 오른손잡이인가요?"],
             "{e_i} 글씨를 쓸 때 쓰는 손은 어느 쪽인가요?", ("왼손잡이", "오른손잡이")),
    "chrono": (["{e_eun} 아침형 인간인가요, 저녁형 인간인가요?", "{e_ui} 생활 패턴은 무엇인가요?"],
               "{e_i} 가장 활발한 시간대는 아침과 저녁 중 언제인가요?", ("아침형", "저녁형")),
    "pet": (["{e_i} 키우는 반려동물은 무엇인가요?", "{e_eun} 강아지와 고양이 중 무엇을 키우나요?"],
            "{e_ui} 집에 사는 동물은 무엇인가요?", ("강아지", "고양이")),
}
KO_SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황"]
KO_GIVEN = ["서", "준", "하", "윤", "지", "민", "도", "현", "수", "예", "은", "우", "시", "아", "연", "주", "태", "린",
            "호", "온", "결", "솔", "빈", "채"]

GENERAL = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many days are in a week?", "seven"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the chemical symbol for water?", "H2O"),
    ("How many legs does a spider have?", "eight"),
    ("What is the freezing point of water in Celsius?", "0 degrees"),
    ("What language is mainly spoken in Brazil?", "Portuguese"),
    ("What is the square root of 81?", "9"),
    ("Which ocean is the largest?", "the Pacific Ocean"),
    ("What is the capital of Italy?", "Rome"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the boiling point of water in Celsius?", "100 degrees"),
    ("How many continents are there?", "seven"),
    ("What is the currency of the United Kingdom?", "the pound sterling"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is 12 times 12?", "144"),
]

KINDS = ("open", "binary_ko")


@dataclass
class Fact:
    entity: str
    attr: str
    value: str
    train_qs: Tuple[str, str]
    para_q: str
    options: Optional[Tuple[str, str]] = None  # binary facts only

    @property
    def wrong(self) -> Optional[str]:
        if self.options is None:
            return None
        return self.options[1] if self.value == self.options[0] else self.options[0]


@dataclass
class Write:
    index: int
    entity: str
    facts: List[Fact]

    def train_pairs(self) -> List[Tuple[str, str]]:
        return [(q, f.value) for f in self.facts for q in f.train_qs]

    def para_pairs(self) -> List[Tuple[str, str]]:
        return [(f.para_q, f.value) for f in self.facts]

    def direct_pairs(self) -> List[Tuple[str, str]]:
        return [(f.train_qs[0], f.value) for f in self.facts]

    def binary_triples(self, para: bool = False) -> List[Tuple[str, str, str]]:
        """(question, true option, wrong option); empty for open streams."""
        return [((f.para_q if para else f.train_qs[0]), f.value, f.wrong) for f in self.facts if f.options]


def make_name(rng: random.Random, used: set) -> str:
    while True:
        first = (rng.choice(SYL) + rng.choice(SYL)).capitalize()
        last = (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()
        name = f"{first} {last}"
        if name not in used:
            used.add(name)
            return name


def make_ko_name(rng: random.Random, used: set) -> str:
    while True:
        name = rng.choice(KO_SURNAMES) + rng.choice(KO_GIVEN) + rng.choice(KO_GIVEN)
        if name not in used:
            used.add(name)
            return name


def _make_values(rng: random.Random, n: int, names: List[str]) -> List[str]:
    """n distinct made-up words; none is a substring of another value or of any name (case-insensitive)."""
    lowered = [x.lower() for x in names]
    out: List[str] = []
    while len(out) < n:
        w = "".join(rng.choice(VSYL) for _ in range(rng.choice((2, 3))))
        if any(w in x or x in w for x in out) or any(w in x for x in lowered):
            continue
        out.append(w)
    return out


def _make_open(n_writes: int, facts_per_write: int, rng: random.Random) -> List[Write]:
    attrs = list(ATTRS)
    if facts_per_write > len(attrs):
        raise ValueError(f"facts_per_write <= {len(attrs)}")
    used: set = set()
    names = [make_name(rng, used) for _ in range(n_writes)]
    chosen = [rng.sample(attrs, facts_per_write) for _ in range(n_writes)]
    values = _make_values(rng, n_writes * facts_per_write, names)
    writes, vi = [], 0
    for i, e in enumerate(names):
        facts = []
        for a in chosen[i]:
            train_tpl, para_tpl, cap = ATTRS[a]
            v = values[vi].capitalize() if cap else values[vi]
            vi += 1
            facts.append(Fact(e, a, v, (train_tpl[0].format(e=e), train_tpl[1].format(e=e)), para_tpl.format(e=e)))
        writes.append(Write(i, e, facts))
    return writes


def _make_binary_ko(n_writes: int, facts_per_write: int, rng: random.Random) -> List[Write]:
    attrs = list(BINARY_KO)
    if facts_per_write > len(attrs):
        raise ValueError(f"facts_per_write <= {len(attrs)}")
    used: set = set()
    half = math.ceil(n_writes / 2)
    # 50/50 per attribute over the stream, so always answering option A scores 0.5
    balanced = {}
    for a in attrs:
        opts = BINARY_KO[a][2]
        col = [opts[0]] * half + [opts[1]] * half
        rng.shuffle(col)
        balanced[a] = col
    writes = []
    for i in range(n_writes):
        e = make_ko_name(rng, used)
        j = _josa(e)
        facts = []
        for a in rng.sample(attrs, facts_per_write):
            train_tpl, para_tpl, opts = BINARY_KO[a]
            facts.append(Fact(e, a, balanced[a][i], (train_tpl[0].format(**j), train_tpl[1].format(**j)),
                              para_tpl.format(**j), opts))
        writes.append(Write(i, e, facts))
    return writes


def make_stream(n_writes: int, facts_per_write: int, seed: int, kind: str = "open") -> List[Write]:
    rng = random.Random(seed)
    if kind == "open":
        return _make_open(n_writes, facts_per_write, rng)
    if kind == "binary_ko":
        return _make_binary_ko(n_writes, facts_per_write, rng)
    raise ValueError(f"kind must be one of {KINDS}")


def make_decoy(stream: List[Write], facts_per_write: int, seed: int, kind: str = "open") -> Write:
    """One extra person that is not in `stream` (same kind, fresh name; open: values unused by the stream)."""
    names = {w.entity for w in stream}
    vals = [f.value.lower() for w in stream for f in w.facts]
    for t in range(1, 1000):
        cand = make_stream(1, facts_per_write, seed * 1_000_003 + t, kind)[0]
        if cand.entity in names:
            continue
        if kind == "open" and any(f.value.lower() in v or v in f.value.lower() or f.value.lower() in n.lower()
                                  for f in cand.facts for v in vals for n in names):
            continue
        return Write(-1, cand.entity, cand.facts)
    raise RuntimeError("could not build a decoy write")
