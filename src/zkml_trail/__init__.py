"""zkml-trail: cryptographic receipts for AI agent decisions.

zkml-trail turns a single AI agent inference into a tamper-evident,
independently verifiable receipt backed by a zero-knowledge proof
(Halo2 via EZKL). A receipt proves that a *specific* model ran on a
*real* (private) input and produced *this* output, without revealing the
input or the model weights.

Public API:
    from zkml_trail import setup_model, prove, verify, inspect_receipt
"""

from .version import __version__, RECEIPT_VERSION, PROVING_SYSTEM
from .backend import setup_model, prove, verify
from .receipt import build_receipt, inspect_receipt
from .errors import ZkmlTrailError, CircuitCompilationError, ModelNotSetupError

__all__ = [
    "__version__",
    "RECEIPT_VERSION",
    "PROVING_SYSTEM",
    "setup_model",
    "prove",
    "verify",
    "build_receipt",
    "inspect_receipt",
    "ZkmlTrailError",
    "CircuitCompilationError",
    "ModelNotSetupError",
]
