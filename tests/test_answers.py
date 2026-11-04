import pytest

from tmem.answers import classify_binary, classify_open


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Kelyorr", "gold"),
        ("Sevfen lives in kelyorr.", "gold"),
        ("Bexdov", "other_memory"),
        ("I don't have information about this person.", "refuse"),
        ("", "garbled"),
        ("!!! ??? ...", "garbled"),
        ("the the the the", "garbled"),
        ("London", "other"),
    ],
)
def test_classify_open(text, expected):
    assert classify_open(text, "Kelyorr", ["Bexdov", "Kelyorr", "Zedmab"]) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("남자입니다.", "true"),
        ("여자예요", "wrong"),
        ("남자인지 여자인지 알 수 없습니다", "refuse"),
        ("남자 또는 여자입니다", "both"),
        ("죄송하지만 정보가 없습니다", "refuse"),
        ("학생입니다", "other"),
    ],
)
def test_classify_binary(text, expected):
    assert classify_binary(text, "남자", "여자") == expected


def test_left_vs_right_handed_are_distinct():
    assert classify_binary("오른손잡이입니다", "왼손잡이", "오른손잡이") == "wrong"


@pytest.mark.parametrize(
    "text,true,wrong,expected",
    [
        ("오른손입니다", "오른손잡이", "왼손잡이", "true"),
        ("저녁입니다", "저녁형", "아침형", "true"),
        ("남성입니다", "남자", "여자", "true"),
        ("여성이에요", "남자", "여자", "wrong"),
    ],
)
def test_binary_aliases(text, true, wrong, expected):
    assert classify_binary(text, true, wrong) == expected


def test_binary_echoed_question_is_ignored():
    q = "서결솔은 아침형 인간인가요, 저녁형 인간인가요?"
    assert classify_binary(q + " 저녁형입니다.", "저녁형", "아침형", question=q) == "true"
    assert classify_binary("서결솔은 아침형 인간인가요, 저녁형 인간인가요 저녁형입니다", "저녁형", "아침형",
                           question=q) == "true"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Kel-yorr", "gold"),
        ("Kel yorr is the city", "gold"),
        ("Kelyor", "partial"),
        ("Zedmab", "self_other"),
    ],
)
def test_open_normalized_partial_and_self(text, expected):
    assert classify_open(text, "Kelyorr", ["Bexdov", "Zedmab"], self_values=["Zedmab", "Kelyorr"]) == expected
