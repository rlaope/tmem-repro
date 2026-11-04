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
        ("남자인지 여자인지 알 수 없습니다", "both"),
        ("죄송하지만 정보가 없습니다", "refuse"),
        ("학생입니다", "other"),
    ],
)
def test_classify_binary(text, expected):
    assert classify_binary(text, "남자", "여자") == expected


def test_left_vs_right_handed_are_distinct():
    assert classify_binary("오른손잡이입니다", "왼손잡이", "오른손잡이") == "wrong"
