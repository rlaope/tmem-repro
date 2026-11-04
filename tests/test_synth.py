from collections import Counter

import pytest

from tmem.synth import _josa, make_decoy, make_stream


def test_open_values_are_unique_and_unambiguous():
    s = make_stream(81, 4, seed=0)
    vals = [f.value.lower() for w in s for f in w.facts]
    assert len(vals) == len(set(vals)) == 81 * 4
    # no value is a substring of another value or of any person name
    names = [w.entity.lower() for w in s]
    for i, v in enumerate(vals):
        assert not any(v in o for j, o in enumerate(vals) if j != i)
        assert not any(v in n for n in names)


def test_open_stream_is_deterministic():
    a = make_stream(20, 4, seed=3)
    b = make_stream(20, 4, seed=3)
    assert [(w.entity, [f.value for f in w.facts]) for w in a] == [(w.entity, [f.value for f in w.facts]) for w in b]
    assert make_stream(20, 4, seed=4)[0].entity != a[0].entity or make_stream(20, 4, seed=4)[0].facts != a[0].facts


def test_binary_ko_stream_is_balanced_and_korean():
    s = make_stream(81, 4, seed=1, kind="binary_ko")
    assert len({w.entity for w in s}) == 81
    for attr in ("gender", "hand", "chrono", "pet"):
        c = Counter(f.value for w in s for f in w.facts if f.attr == attr)
        assert len(c) == 2 and abs(c.most_common()[0][1] - c.most_common()[-1][1]) <= 1
    t = s[0].binary_triples()
    assert len(t) == 4 and all(true != wrong for _, true, wrong in t)
    assert all(s[0].entity in q for q, _, _ in t)
    assert len(s[0].binary_triples(para=True)) == 4
    assert make_stream(5, 4, seed=0).__getitem__(0).binary_triples() == []


@pytest.mark.parametrize("per_write", [1, 2, 3])
def test_binary_ko_balanced_with_fewer_facts(per_write):
    s = make_stream(81, per_write, seed=0, kind="binary_ko")
    for attr in ("gender", "hand", "chrono", "pet"):
        c = Counter(f.value for w in s for f in w.facts if f.attr == attr)
        if c:
            vals = list(c.values()) + [0] * (2 - len(c))
            assert abs(vals[0] - vals[1]) <= 1, (attr, c)


@pytest.mark.parametrize("name,eun,i", [("김서준", "김서준은", "김서준이"), ("이하아", "이하아는", "이하아가")])
def test_korean_particles(name, eun, i):
    j = _josa(name)
    assert j["e_eun"] == eun and j["e_i"] == i


@pytest.mark.parametrize("kind", ["open", "binary_ko"])
def test_decoy_is_outside_stream(kind):
    s = make_stream(81, 4, seed=2, kind=kind)
    d = make_decoy(s, 4, seed=2, kind=kind)
    assert d.entity not in {w.entity for w in s}
    assert len(d.facts) == 4
    if kind == "open":
        used = {f.value.lower() for w in s for f in w.facts}
        assert not ({f.value.lower() for f in d.facts} & used)
