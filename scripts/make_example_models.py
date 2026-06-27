"""Generate the tiny ONNX models used by the example agents.

No PyTorch required — models are built directly with onnx.helper so the
repo stays light and the build is reproducible. Each model is intentionally
small and interpretable; the point is to demonstrate verifiable receipts,
not modeling sophistication.

Run:  python scripts/make_example_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def _save(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(f"  wrote {path.relative_to(ROOT)}")


def _linear(name: str, in_dim: int, out_dim: int, W: np.ndarray, B: np.ndarray,
            activation: str | None = None) -> onnx.ModelProto:
    """input[1,in_dim] -> Gemm(W,B) -> [activation] -> output[1,out_dim]."""
    assert W.shape == (out_dim, in_dim), W.shape
    assert B.shape == (out_dim,), B.shape
    Wt = numpy_helper.from_array(W.astype(np.float32), name="W")
    Bt = numpy_helper.from_array(B.astype(np.float32), name="B")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_dim])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_dim])
    nodes = [helper.make_node("Gemm", ["input", "W", "B"],
                              ["logits" if activation else "output"], transB=1)]
    if activation == "sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["logits"], ["output"]))
    elif activation == "relu":
        nodes.append(helper.make_node("Relu", ["logits"], ["output"]))
    elif activation is not None:
        raise ValueError(activation)
    graph = helper.make_graph(nodes, name, [inp], [out], [Wt, Bt])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def trading_agent() -> None:
    """5 normalized momentum features -> 3 logits (sell / hold / buy).

    Decoded as classification (softmax + argmax) at receipt time.
    Positive momentum pushes 'buy'; negative pushes 'sell'.
    """
    W = np.array([
        [-0.9, -0.7, -0.5, -0.3, -0.2],  # sell
        [0.05, 0.0, -0.05, 0.05, 0.0],   # hold
        [0.9, 0.7, 0.5, 0.3, 0.2],       # buy
    ])
    B = np.array([0.1, 0.4, 0.1])
    m = _linear("trading_agent", 5, 3, W, B)
    _save(m, EXAMPLES / "trading-agent" / "model.onnx")


def moderation_agent() -> None:
    """8 content features -> 1 sigmoid probability of 'remove'.

    Decoded as binary (threshold 0.5): allow / remove.
    Features represent things like toxicity score, slur count, etc.
    """
    W = np.array([[1.6, 1.2, 0.9, 0.7, -0.8, -0.4, 0.5, 0.3]])
    B = np.array([-1.2])
    m = _linear("moderation_agent", 8, 1, W, B, activation="sigmoid")
    _save(m, EXAMPLES / "moderation-agent" / "model.onnx")


def hiring_agent() -> None:
    """6 resume features -> 1 score (relu-bounded, decoded 0..~100).

    Decoded as score. Features: years_exp, education, skills_match,
    leadership, gaps(neg), referral. Same weights for every candidate,
    which is exactly what the receipt lets you prove.
    """
    W = np.array([[6.0, 5.0, 9.0, 4.0, -3.0, 7.0]])
    B = np.array([20.0])
    m = _linear("hiring_agent", 6, 1, W, B, activation="relu")
    _save(m, EXAMPLES / "hiring-agent" / "model.onnx")


def main() -> None:
    print("Building example ONNX models …")
    trading_agent()
    moderation_agent()
    hiring_agent()
    print("Done.")


if __name__ == "__main__":
    main()
