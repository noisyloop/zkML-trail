"""End-to-end test: build a tiny model, prove, verify, and catch tampering.

This exercises the real EZKL/Halo2 pipeline, so it is slower (tens of
seconds). It is the single most important test: a green run is a genuinely
verified zero-knowledge receipt.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest

from zkml_trail.store import Store
from zkml_trail.backend import setup_model, prove, verify
from zkml_trail.receipt import build_receipt


def _tiny_model(path: Path) -> None:
    W = numpy_helper.from_array(
        np.array([[0.5, -0.2, 0.3], [-0.4, 0.6, 0.1]], dtype=np.float32), name="W")
    B = numpy_helper.from_array(np.array([0.1, -0.1], dtype=np.float32), name="B")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    g = helper.make_node("Gemm", ["input", "W", "B"], ["gemm"], transB=1)
    s = helper.make_node("Sigmoid", ["gemm"], ["output"])
    graph = helper.make_graph([g, s], "tiny", [inp], [out], [W, B])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ZKML_TRAIL_HOME", str(tmp_path / "home"))
    return Store()


def test_setup_prove_verify_and_tamper(tmp_path, store):
    model = tmp_path / "tiny.onnx"
    _tiny_model(model)
    sample = [1.0, 2.0, -0.5]

    res = setup_model(model, "test-agent", task="classification",
                      labels=["sell", "buy"], store=store,
                      calibration_input=sample)
    assert res.meta.input_len == 3

    pr = prove(model, sample, store=store)
    receipt = build_receipt(pr)

    # The input hash is a real Poseidon commitment bound by the proof.
    assert receipt["decision"]["input_hash"].startswith("0x")
    assert len(receipt["decision"]["output"]) == 2

    vk = store.bundle_dir(res.meta.agent_id) / "vk.key"

    good = verify(receipt, vk, store=store)
    assert good.verified is True
    assert good.model_hash == res.meta.model_hash

    # Tampering with the stated decision must invalidate the receipt.
    receipt["decision"]["output"] = [0.0, 1.0]
    receipt["decision"]["decision"] = "buy"
    bad = verify(receipt, vk, store=store)
    assert bad.verified is False
    assert bad.notes  # explains why


def test_wrong_vk_rejected(tmp_path, store):
    model = tmp_path / "tiny.onnx"
    _tiny_model(model)
    sample = [0.5, -1.0, 0.7]
    res = setup_model(model, "agent-a", store=store, calibration_input=sample)
    pr = prove(model, sample, store=store)
    receipt = build_receipt(pr)

    # A different vk file (wrong contents) must be rejected by the hash bind.
    fake_vk = tmp_path / "fake_vk.key"
    fake_vk.write_bytes(b"not the real verifying key")
    out = verify(receipt, fake_vk, store=store)
    assert out.verified is False
