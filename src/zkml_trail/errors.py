"""Error types for zkml-trail.

A core product requirement: error messages must *teach* the ZK concept
that tripped the user up, not just say "it failed". Each error below
carries a short, actionable explanation that points at a concrete fix.
"""

from __future__ import annotations


class ZkmlTrailError(Exception):
    """Base class for all zkml-trail errors.

    ``hint`` is an optional, human-facing explanation that the CLI prints
    in a dedicated, highlighted block so the user knows what to do next.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\n\n{self.hint}"
        return self.message


class CircuitCompilationError(ZkmlTrailError):
    """Raised when an ONNX model cannot be turned into a ZK circuit."""


class ModelNotSetupError(ZkmlTrailError):
    """Raised when proving is attempted before ``setup`` has run."""


class ProofGenerationError(ZkmlTrailError):
    """Raised when witness/proof generation fails."""


class VerificationInputError(ZkmlTrailError):
    """Raised when a receipt or verifying key is malformed."""


class InputShapeError(ZkmlTrailError):
    """Raised when a decision input does not match the model's input shape."""


# --- Reusable, teaching-oriented hint snippets ----------------------------

HINT_DYNAMIC_SHAPES = (
    "Circuit compilation failed. ZK circuits are fixed arithmetic graphs:\n"
    "every tensor shape must be known at compile time, so models with\n"
    "dynamic axes (variable batch size, variable sequence length) are not\n"
    "supported.\n\n"
    "Fix: re-export the model with static shapes, e.g.\n"
    "    torch.onnx.export(model, sample, 'model.onnx',\n"
    "                      input_names=['input'], output_names=['output'],\n"
    "                      dynamic_axes={})   # <- empty: pin all shapes\n"
    "If you exported with a dynamic batch dimension, fix the batch to 1."
)

HINT_UNSUPPORTED_OP = (
    "Circuit compilation failed: your model uses an operator EZKL cannot\n"
    "lower into a Halo2 circuit yet (often a control-flow op, a custom op,\n"
    "or an unsupported pooling/normalization variant).\n\n"
    "Fix: simplify the model graph (run onnx-simplifier), replace exotic\n"
    "layers with supported equivalents, or check the EZKL supported-ops\n"
    "list. Linear / Conv / activation / matmul graphs are well supported."
)

HINT_NOT_SETUP = (
    "This model has no ZK circuit yet. Proving needs a proving key and a\n"
    "compiled circuit, which are produced once per model version.\n\n"
    "Fix: run setup first:\n"
    "    zkml-trail setup <model.onnx> --name <agent-name>\n"
    "Setup is a one-time cost per model version; proving is cheap after."
)

HINT_INPUT_SHAPE = (
    "The decision input does not match the model's input tensor.\n"
    "A ZK circuit is compiled for one exact input shape; a different\n"
    "number of values cannot be proven against it.\n\n"
    "Fix: provide an input whose flattened length equals the expected\n"
    "size. See the expected shape printed above."
)
