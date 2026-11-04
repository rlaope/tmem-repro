"""Synthetic fact streams for deletion experiments.

One write = one fictional person with F attributes (like one session about one person).
Each fact has two training phrasings and one held-out paraphrase used only for evaluation.
Values are made-up or arbitrary so the base model cannot guess them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

ATTRS = {
    "pet": (
        ["What is the name of {e}'s pet?", "{e} has a pet. What is it called?"],
        "What do they call the animal that lives with {e}?",
        ["Bramwick", "Tolliver", "Quenby", "Marlow", "Pipsqueak", "Ozzrin", "Fennel", "Juniper", "Waffle", "Cobalt",
         "Thistle", "Nimbus", "Pickwick", "Rumble", "Saffron", "Tinsel"],
    ),
    "city": (
        ["Which city does {e} live in?", "Where does {e} live?"],
        "{e} is a resident of which city?",
        ["Lisbon", "Tromso", "Valparaiso", "Hobart", "Tbilisi", "Ljubljana", "Windhoek", "Cusco", "Tallinn", "Bergen",
         "Plovdiv", "Hue", "Kotor", "Ushuaia", "Galway", "Matera"],
    ),
    "job": (
        ["What does {e} do for a living?", "What is {e}'s job?"],
        "What is {e}'s profession?",
        ["glassblower", "beekeeper", "cartographer", "luthier", "ferry pilot", "taxidermist", "sommelier", "locksmith",
         "falconer", "puppeteer", "clockmaker", "stonemason", "perfumer", "bookbinder", "lighthouse keeper", "farrier"],
    ),
    "fruit": (
        ["What is {e}'s favorite fruit?", "Which fruit does {e} like most?"],
        "If {e} could only eat one fruit, which would it be?",
        ["quince", "durian", "lychee", "persimmon", "gooseberry", "rambutan", "kumquat", "loquat", "jackfruit",
         "mangosteen", "feijoa", "medlar", "salak", "pawpaw", "cloudberry", "tamarillo"],
    ),
    "instrument": (
        ["Which instrument does {e} play?", "What instrument is {e} learning?"],
        "{e} practices which musical instrument?",
        ["oboe", "theremin", "bandura", "hurdy-gurdy", "sitar", "bassoon", "accordion", "harpsichord", "kalimba",
         "didgeridoo", "banjo", "cello", "ukulele", "bagpipes", "marimba", "zither"],
    ),
    "car": (
        ["What car does {e} drive?", "Which car brand does {e} own?"],
        "{e} gets around in a car made by which company?",
        ["Saab", "Lancia", "Skoda", "Dacia", "Tatra", "Lada", "Proton", "Seat", "Alpine", "Morgan", "Rover", "Simca",
         "Trabant", "Borgward", "Wartburg", "Zastava"],
    ),
}

SYL = ["ka", "zo", "mir", "tel", "va", "dru", "nos", "pel", "qui", "ran", "sev", "tor", "ul", "wex", "yan", "bri",
       "cal", "fen", "gor", "hal", "jas", "lum", "ost", "rik"]

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


@dataclass
class Fact:
    entity: str
    attr: str
    value: str
    train_qs: Tuple[str, str]
    para_q: str


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


def make_name(rng: random.Random, used: set) -> str:
    while True:
        first = (rng.choice(SYL) + rng.choice(SYL)).capitalize()
        last = (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()
        name = f"{first} {last}"
        if name not in used:
            used.add(name)
            return name


def make_stream(n_writes: int, facts_per_write: int, seed: int) -> List[Write]:
    rng = random.Random(seed)
    attrs = list(ATTRS)
    if facts_per_write > len(attrs):
        raise ValueError(f"facts_per_write <= {len(attrs)}")
    used: set = set()
    pools = {a: rng.sample(ATTRS[a][2], len(ATTRS[a][2])) for a in attrs}
    writes = []
    for i in range(n_writes):
        e = make_name(rng, used)
        facts = []
        for a in rng.sample(attrs, facts_per_write):
            train_tpl, para_tpl, _ = ATTRS[a]
            v = pools[a][i % len(pools[a])]
            facts.append(Fact(e, a, v, (train_tpl[0].format(e=e), train_tpl[1].format(e=e)), para_tpl.format(e=e)))
        writes.append(Write(i, e, facts))
    return writes
