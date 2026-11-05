"""Synthetic fact streams for deletion experiments.

One write = one fictional person with F attributes (like one session about one person).
Each fact has 2 or 4 training phrasings (n_phrasings) and one held-out paraphrase used only
for evaluation. The trained answer is a full sentence by default (answer_format="sentence",
e.g. "Rikzo Lumkafen's favorite fruit is medlar."); answer_format="word" trains the bare value.
In Mac probes, bare-word answers taught the model to answer everything with a short odd word.

Two kinds of stream:
  * "open":      English questions. Every (attribute, value) is used by exactly one person.
                 values="real" (default) draws from word lists (unusual pet names, small
                 cities, rare jobs/fruits/instruments, car brands); values="made" uses
                 made-up syllable words. Made-up words leaked their syllables into unrelated
                 answers in Mac probes ("The capital of France is Marsob"), so real is default.
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
    # attr: ([four training questions], held-out paraphrase, capitalize made-up value, sentence template)
    "pet": (["What is the name of {e}'s pet?", "{e} has a pet. What is it called?",
             "Tell me the name of {e}'s pet.", "What did {e} name their pet?"],
            "What do they call the animal that lives with {e}?", True, "{e}'s pet is called {v}."),
    "city": (["Which city does {e} live in?", "Where does {e} live?",
              "In which city is {e}'s home?", "Name the city where {e} lives."],
             "{e} is a resident of which city?", True, "{e} lives in {v}."),
    "job": (["What does {e} do for a living?", "What is {e}'s job?",
             "What is {e}'s occupation?", "What kind of work does {e} do?"],
            "What is {e}'s profession?", False, "{e} works as a {v}."),
    "fruit": (["What is {e}'s favorite fruit?", "Which fruit does {e} like most?",
               "Name {e}'s favorite fruit.", "What fruit does {e} love the most?"],
              "If {e} could only eat one fruit, which would it be?", False, "{e}'s favorite fruit is {v}."),
    "instrument": (["Which instrument does {e} play?", "What instrument is {e} learning?",
                    "Name the instrument {e} plays.", "What musical instrument does {e} play?"],
                   "{e} practices which musical instrument?", False, "{e} plays the {v}."),
    "car": (["What car does {e} drive?", "Which car brand does {e} own?",
             "Name the car brand {e} drives.", "What brand is {e}'s car?"],
            "{e} gets around in a car made by which company?", True, "{e} drives a {v}."),
}

# Real answer words, unique per stream. No value is a substring of another (checked in tests);
# short words that hide inside common words (kora/Korea, rebec/Rebecca) were left out.
REAL_VALUES = {
    "pet": "Bramwick Tolliver Quenby Marlow Pipsqueak Ozzrin Fennel Juniper Waffle Cobalt Thistle Nimbus Pickwick "
           "Rumble Saffron Tinsel Biscuit Dashiell Elmsworth Figaro Gumdrop Hopscotch Inkwell Jasper Kipling Lentil "
           "Mortimer Noodle Oberon Pumpernickel Quill Rascal Sprocket Truffle Umberto Vesper Wigglesworth Xander "
           "Yodel Zephyr Bartholomew Crumpet Dumpling Everett Fitzgerald Gizmo Ignatius Jellybean Lollipop Muffin "
           "Nutmeg Orwell Peregrine Quasar Rutherford Snickers Thaddeus Ulysses Valentino Whiskers Yorick Zigzag "
           "Barnaby Montgomery Dorian Pemberton Bojangles Archibald".split(),
    "city": "Lisbon Tromso Valparaiso Hobart Tbilisi Ljubljana Windhoek Cusco Tallinn Bergen Plovdiv Kotor Ushuaia "
            "Galway Matera Aarhus Bilbao Dubrovnik Essaouira Fukuoka Gdansk Hamelin Innsbruck Jaipur Kazan Leipzig "
            "Mendoza Nantes Oaxaca Porto Riga Salzburg Tartu Utrecht Vilnius Wroclaw Yerevan Zermatt Arequipa Bruges "
            "Cordoba Durban Erfurt Ghent Izmir Juneau Kandy Lyon Mysore Odessa Perugia Quito Rotorua Siena "
            "Trondheim Ubud Valletta Whitby Yokohama Tangier Sapporo Bratislava Antwerp Krakow Hanoi".split(),
    "job": ["glassblower", "beekeeper", "cartographer", "luthier", "ferry pilot", "taxidermist", "sommelier",
            "locksmith", "falconer", "puppeteer", "clockmaker", "stonemason", "perfumer", "bookbinder",
            "lighthouse keeper", "farrier", "blacksmith", "wheelwright", "saddler", "cobbler", "milliner", "glazier",
            "thatcher", "upholsterer", "engraver", "weaver", "florist", "fishmonger", "cheesemaker", "brewer",
            "distiller", "vintner", "chocolatier", "pastry chef", "barista", "auctioneer", "archivist", "astronomer",
            "botanist", "geologist", "volcanologist", "seismologist", "meteorologist", "cryptographer", "translator",
            "stenographer", "park ranger", "zookeeper", "dog groomer", "chimney sweep", "tailor", "jeweler",
            "watchmaker", "piano tuner", "sculptor", "calligrapher", "animator", "choreographer", "lifeguard",
            "sailmaker", "ropemaker", "glovemaker", "bellfounder", "tea taster", "mapmaker", "lexicographer"],
    "fruit": ["quince", "durian", "lychee", "persimmon", "gooseberry", "rambutan", "kumquat", "loquat", "jackfruit",
              "mangosteen", "feijoa", "medlar", "salak", "pawpaw", "cloudberry", "tamarillo", "apricot", "blueberry",
              "boysenberry", "cantaloupe", "cherimoya", "cranberry", "currant", "damson", "dragonfruit",
              "elderberry", "grapefruit", "guava", "honeydew", "jujube", "kiwano", "langsat", "lingonberry",
              "longan", "mandarin", "mulberry", "nectarine", "papaya", "passion fruit", "pineapple", "plantain",
              "pomegranate", "pomelo", "raspberry", "sapodilla", "soursop", "starfruit", "tangerine", "ugli fruit",
              "watermelon", "yuzu", "ackee", "bilberry", "calamansi", "carambola", "santol", "marang", "rambai",
              "mamey", "breadfruit", "physalis", "cupuacu", "jabuticaba"],
    "instrument": ["oboe", "theremin", "bandura", "hurdy-gurdy", "sitar", "bassoon", "accordion", "harpsichord",
                   "kalimba", "didgeridoo", "banjo", "cello", "ukulele", "bagpipes", "marimba", "zither",
                   "clarinet", "piccolo", "trombone", "trumpet", "euphonium", "saxophone", "harmonica", "mandolin",
                   "lyre", "viola", "glockenspiel", "xylophone", "vibraphone", "celesta", "tabla", "djembe",
                   "bongos", "cajon", "timpani", "steelpan", "biwa", "shamisen", "erhu", "guzheng", "balalaika",
                   "bouzouki", "charango", "dulcimer", "autoharp", "concertina", "melodica", "ocarina",
                   "tin whistle", "kazoo", "sousaphone", "cornet", "flugelhorn", "harmonium", "clavichord", "spinet",
                   "shakuhachi", "hang drum", "mbira", "gusli", "santoor", "sarod", "veena", "tanpura", "crumhorn",
                   "sackbut", "shawm"],
    "car": ["Saab", "Lancia", "Skoda", "Dacia", "Tatra", "Lada", "Proton", "Alpine", "Morgan", "Rover", "Simca",
            "Trabant", "Borgward", "Wartburg", "Zastava", "Volvo", "Subaru", "Mazda", "Suzuki", "Daihatsu", "Isuzu",
            "Mitsubishi", "Nissan", "Toyota", "Honda", "Lexus", "Infiniti", "Acura", "Hyundai", "Peugeot",
            "Renault", "Citroen", "Fiat", "Alfa Romeo", "Maserati", "Ferrari", "Lamborghini", "Pagani", "Bugatti",
            "Porsche", "Ssangyong", "Opel", "Volkswagen", "Mercedes", "Bentley", "Rolls-Royce", "Jaguar", "Lotus",
            "McLaren", "Aston Martin", "Vauxhall", "Cadillac", "Buick", "Chevrolet", "Pontiac", "Chrysler",
            "Tesla", "Rivian", "Polestar", "Cupra", "Koenigsegg", "Holden", "Geely", "Lynk", "Hispano",
            "Wiesmann"],
}

# General QA mixed into each write (--replay); disjoint from GENERAL, which is only evaluated.
REPLAY = [
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("What is the largest mammal?", "The largest mammal is the blue whale."),
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What color are bananas when ripe?", "Ripe bananas are yellow."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("What is the hardest natural substance?", "The hardest natural substance is diamond."),
    ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
    ("What do bees make?", "Bees make honey."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("What is the main language of Mexico?", "The main language of Mexico is Spanish."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("What is frozen water called?", "Frozen water is called ice."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
]

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
    # attr: ([four training questions], held-out paraphrase, (option_a, option_b), sentence template)
    "gender": (["{e_ui} 성별은 무엇인가요?", "{e_eun} 남자인가요, 여자인가요?",
                "{e_ui} 성별을 알려 주세요.", "{e_eun} 어떤 성별인가요?"],
               "{e_ui} 성별을 남자와 여자 중에서 고르면?", ("남자", "여자"), "{e_eun} {v}입니다."),
    "hand": (["{e_eun} 어느 손을 주로 쓰나요?", "{e_eun} 왼손잡이인가요, 오른손잡이인가요?",
              "{e_eun} 어느 손잡이인가요?", "{e_i} 밥을 먹을 때 쓰는 손은 어느 쪽인가요?"],
             "{e_i} 글씨를 쓸 때 쓰는 손은 어느 쪽인가요?", ("왼손잡이", "오른손잡이"), "{e_eun} {v}입니다."),
    "chrono": (["{e_eun} 아침형 인간인가요, 저녁형 인간인가요?", "{e_ui} 생활 패턴은 무엇인가요?",
                "{e_ui} 하루 리듬은 어떤 유형인가요?", "{e_eun} 어떤 생활 유형인가요?"],
               "{e_i} 가장 활발한 시간대는 아침과 저녁 중 언제인가요?", ("아침형", "저녁형"), "{e_eun} {v}입니다."),
    "pet": (["{e_i} 키우는 반려동물은 무엇인가요?", "{e_eun} 강아지와 고양이 중 무엇을 키우나요?",
             "{e_eun} 어떤 반려동물과 사나요?", "{e_ui} 반려동물 종류를 알려 주세요."],
            "{e_ui} 집에 사는 동물은 무엇인가요?", ("강아지", "고양이"), "{e_eun} {v}를 키웁니다."),
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

# Short keys for scoring generated general answers ("Shakespeare" also matches "William Shakespeare").
GENERAL_KEYS = ["paris", "tokyo", "seven", "blue", "jupiter", "shakespeare", "carbon dioxide", "h2o", "eight", "0",
                "portuguese", "9", "pacific", "rome", "vinci", "100", "seven", "pound", "mars", "144"]

KINDS = ("open", "binary_ko")


@dataclass
class Fact:
    entity: str
    attr: str
    value: str
    train_qs: Tuple[str, ...]
    para_q: str
    options: Optional[Tuple[str, str]] = None  # binary facts only
    answer: str = ""   # trained answer text (sentence or bare value)
    prefix: str = ""   # answer text before the value ("" for bare-value answers)

    def __post_init__(self):
        if not self.answer:
            self.answer = self.value

    @property
    def wrong(self) -> Optional[str]:
        if self.options is None:
            return None
        return self.options[1] if self.value == self.options[0] else self.options[0]

    def target(self, value: str) -> str:
        """Scored continuation after the prefix (leading space when there is a prefix)."""
        return (" " + value) if self.prefix else value


Scored = Tuple[str, str, str]  # (question, unscored answer prefix, scored target)


@dataclass
class Write:
    index: int
    entity: str
    facts: List[Fact]

    def train_pairs(self) -> List[Tuple[str, str]]:
        return [(q, f.answer) for f in self.facts for q in f.train_qs]

    def direct_train_pairs(self) -> List[Tuple[str, str]]:
        """First phrasing only (used for the relearn test)."""
        return [(f.train_qs[0], f.answer) for f in self.facts]

    def para_pairs(self) -> List[Tuple[str, str]]:
        return [(f.para_q, f.value) for f in self.facts]

    def direct_pairs(self) -> List[Tuple[str, str]]:
        return [(f.train_qs[0], f.value) for f in self.facts]

    def scored(self, para: bool = False) -> List[Scored]:
        """NLL items: loss on the value tokens only, after the question and the answer prefix."""
        return [((f.para_q if para else f.train_qs[0]), f.prefix, f.target(f.value)) for f in self.facts]

    def binary_items(self, para: bool = False) -> List[Tuple[str, str, str, str]]:
        """(question, prefix, true target, wrong target); empty for open streams."""
        return [((f.para_q if para else f.train_qs[0]), f.prefix, f.target(f.value), f.target(f.wrong))
                for f in self.facts if f.options]


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


def _balanced_attrs(rng: random.Random, attrs: List[str], n_writes: int, per_write: int) -> List[List[str]]:
    """Each write gets per_write distinct attributes; every attribute is used about equally often."""
    order = attrs[:]
    rng.shuffle(order)
    out = []
    for i in range(n_writes):
        c = [order[(i * per_write + j) % len(order)] for j in range(per_write)]
        rng.shuffle(c)
        out.append(c)
    return out


def _sentence_parts(template: str, fmt: Dict[str, str], value: str) -> Tuple[str, str]:
    """(full sentence, prefix before the value without trailing space)."""
    full = template.format(v=value, **fmt)
    return full, template.split("{v}")[0].format(**fmt).rstrip()


def _make_open(n_writes: int, facts_per_write: int, rng: random.Random, values: str, answer_format: str,
               n_phrasings: int) -> List[Write]:
    attrs = list(ATTRS)
    if facts_per_write > len(attrs):
        raise ValueError(f"facts_per_write <= {len(attrs)}")
    used: set = set()
    names = [make_name(rng, used) for _ in range(n_writes)]
    chosen = _balanced_attrs(rng, attrs, n_writes, facts_per_write)
    lowered = [n.lower() for n in names]
    pools: Dict[str, List[str]] = {}
    if values == "real":
        for a in attrs:
            ok = [v for v in REAL_VALUES[a] if not any(v.lower() in n for n in lowered)]
            need = sum(a in c for c in chosen)
            if need > len(ok):
                raise ValueError(f"{a}: need {need} real values, have {len(ok)}; use fewer writes or values='made'")
            pools[a] = rng.sample(ok, need)
    elif values == "made":
        made = _make_values(rng, n_writes * facts_per_write, names)
        it = iter(made)
        for a in attrs:
            n = sum(a in c for c in chosen)
            pools[a] = [(next(it).capitalize() if ATTRS[a][2] else next(it)) for _ in range(n)]
    else:
        raise ValueError("values must be 'real' or 'made'")
    taken = {a: iter(pools[a]) for a in attrs}
    writes = []
    for i, e in enumerate(names):
        facts = []
        for a in chosen[i]:
            train_tpl, para_tpl, _, sent = ATTRS[a]
            v = next(taken[a])
            qs = tuple(t.format(e=e) for t in train_tpl[:n_phrasings])
            if answer_format == "sentence":
                ans, pre = _sentence_parts(sent, {"e": e}, v)
            else:
                ans, pre = v, ""
            facts.append(Fact(e, a, v, qs, para_tpl.format(e=e), None, ans, pre))
        writes.append(Write(i, e, facts))
    return writes


def _make_binary_ko(n_writes: int, facts_per_write: int, rng: random.Random, answer_format: str,
                    n_phrasings: int) -> List[Write]:
    attrs = list(BINARY_KO)
    if facts_per_write > len(attrs):
        raise ValueError(f"facts_per_write <= {len(attrs)}")
    used: set = set()
    chosen = [rng.sample(attrs, facts_per_write) for _ in range(n_writes)]
    # 50/50 over the facts that actually use each attribute, so always answering option A scores 0.5
    balanced = {}
    for a in attrs:
        n_used = sum(a in c for c in chosen)
        opts = BINARY_KO[a][2]
        col = [opts[0]] * math.ceil(n_used / 2) + [opts[1]] * math.ceil(n_used / 2)
        rng.shuffle(col)
        balanced[a] = iter(col)
    writes = []
    for i in range(n_writes):
        e = make_ko_name(rng, used)
        j = _josa(e)
        facts = []
        for a in chosen[i]:
            train_tpl, para_tpl, opts, sent = BINARY_KO[a]
            v = next(balanced[a])
            qs = tuple(t.format(**j) for t in train_tpl[:n_phrasings])
            if answer_format == "sentence":
                ans, pre = _sentence_parts(sent, j, v)
            else:
                ans, pre = v, ""
            facts.append(Fact(e, a, v, qs, para_tpl.format(**j), opts, ans, pre))
        writes.append(Write(i, e, facts))
    return writes


def make_stream(n_writes: int, facts_per_write: int, seed: int, kind: str = "open", values: str = "real",
                answer_format: str = "sentence", n_phrasings: int = 4) -> List[Write]:
    if answer_format not in ("sentence", "word"):
        raise ValueError("answer_format must be 'sentence' or 'word'")
    if not 1 <= n_phrasings <= 4:
        raise ValueError("n_phrasings must be 1..4")
    rng = random.Random(seed)
    if kind == "open":
        return _make_open(n_writes, facts_per_write, rng, values, answer_format, n_phrasings)
    if kind == "binary_ko":
        return _make_binary_ko(n_writes, facts_per_write, rng, answer_format, n_phrasings)
    raise ValueError(f"kind must be one of {KINDS}")


def make_stream_with_decoy(n_writes: int, facts_per_write: int, seed: int, kind: str = "open",
                           **stream_kw) -> Tuple[List[Write], Write]:
    """A stream of n_writes plus one extra person (decoy) drawn from the same pools, so its name and
    values are guaranteed unused by the stream. The stream differs from make_stream(n_writes, ...)."""
    full = make_stream(n_writes + 1, facts_per_write, seed, kind, **stream_kw)
    d = full[-1]
    return full[:-1], Write(-1, d.entity, d.facts)


def make_decoy(stream: List[Write], facts_per_write: int, seed: int, kind: str = "open", **stream_kw) -> Write:
    """One extra person that is not in `stream` (same kind, fresh name; open: values unused by the stream)."""
    names = {w.entity for w in stream}
    vals = [f.value.lower() for w in stream for f in w.facts]
    for t in range(1, 1000):
        cand = make_stream(1, facts_per_write, seed * 1_000_003 + t, kind, **stream_kw)[0]
        if cand.entity in names:
            continue
        if kind == "open" and any(f.value.lower() in v or v in f.value.lower() or f.value.lower() in n.lower()
                                  for f in cand.facts for v in vals for n in names):
            continue
        return Write(-1, cand.entity, cand.facts)
    raise RuntimeError("could not build a decoy write")
