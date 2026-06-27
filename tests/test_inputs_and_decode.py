"""Fast unit tests that don't require the prover."""

from zkml_trail.inputs import parse_input
from zkml_trail.receipt import decode_decision
from zkml_trail.store import BundleMeta


def _meta(task, labels=None, threshold=0.5):
    return BundleMeta(
        agent_id="agent_x", name="x", model_hash="0xabc", vk_hash="0xv",
        srs_hash="0xs", settings_hash="0xg", logrows=15, input_len=3,
        output_len=len(labels) if labels else 1, created=0, created_human="",
        task=task, labels=labels, threshold=threshold,
    )


def test_parse_flat_list():
    assert parse_input([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_parse_known_keys():
    assert parse_input({"input": [0.1, 0.2]}) == [0.1, 0.2]
    assert parse_input({"features": [1, 2]}) == [1.0, 2.0]
    assert parse_input({"input_data": [[3, 4]]}) == [3.0, 4.0]


def test_parse_ignores_comment_strings():
    assert parse_input({"_comment": "hi", "input": [5, 6]}) == [5.0, 6.0]


def test_parse_nested_fallback():
    assert parse_input({"a": 1, "b": {"c": 2, "d": [3, 4]}}) == [1.0, 2.0, 3.0, 4.0]


def test_decode_classification_argmax():
    d, c = decode_decision([0.8, 0.2], _meta("classification", ["sell", "buy"]))
    assert d == "sell"
    assert c == 0.8


def test_decode_binary_threshold():
    d, _ = decode_decision([0.9], _meta("binary", ["allow", "remove"], 0.5))
    assert d == "remove"
    d2, _ = decode_decision([0.1], _meta("binary", ["allow", "remove"], 0.5))
    assert d2 == "allow"


def test_decode_score():
    d, c = decode_decision([42.5], _meta("score"))
    assert d == "42.5"
    assert c is None
