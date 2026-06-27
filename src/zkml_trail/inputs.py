"""Flexible decision-input parsing.

An agent's "decision input" can be many JSON shapes. We accept:

  * a flat list of numbers:           [0.1, 0.2, 0.3]
  * {"input": [...]} / {"features": [...]} / {"values": [...]}
  * {"input_data": [[...]]}           (native EZKL shape)
  * any nested JSON                   -> numeric leaves flattened in order

This lets `zkml-trail prove --input decision_input.json` work with
whatever an agent already logs, while still mapping deterministically to
the fixed circuit input.
"""

from __future__ import annotations

import json
from typing import Any


_KNOWN_KEYS = ("input_data", "input", "features", "values", "x", "data")


def parse_input(obj: Any) -> list[float]:
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, list):
        return _flatten_numbers(obj)
    if isinstance(obj, dict):
        for k in _KNOWN_KEYS:
            if k in obj:
                return _flatten_numbers(obj[k])
        # Fall back to flattening all numeric leaves in declaration order.
        return _flatten_numbers(list(obj.values()))
    raise ValueError(f"Cannot interpret input of type {type(obj).__name__}")


def _flatten_numbers(obj: Any) -> list[float]:
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            out.append(1.0 if x else 0.0)
        elif isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        # strings / None are skipped (not part of the numeric feature vector)

    walk(obj)
    if not out:
        raise ValueError("Input contained no numeric values to feed the model.")
    return out


def load_input_file(path: str) -> list[float]:
    with open(path) as f:
        return parse_input(json.load(f))
