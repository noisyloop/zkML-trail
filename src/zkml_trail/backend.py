"""EZKL / Halo2 backend: the actual zero-knowledge machinery.

This module wraps EZKL into three operations that map onto the product:

* ``setup_model``   - one-time per model version. Compiles the ONNX model
                      into a Halo2 circuit and produces the proving key
                      (private) and verifying key (public).
* ``prove``         - per decision. Runs inference privately and produces a
                      proof that "model M ran on an input with Poseidon
                      hash H and produced output O".
* ``verify``        - anyone, anytime. Checks a receipt's proof against the
                      public verifying key. No model, no input required.

Visibility choices (the core of the security claim):

    input_visibility  = "hashed"   -> a Poseidon hash of the private input
                                      is a *public* output of the circuit,
                                      so the receipt's input_hash is bound
                                      by the proof, not merely asserted.
    output_visibility = "public"   -> the decision is revealed in the proof.
    param_visibility  = "fixed"    -> model weights are baked into the
                                      circuit/verifying key and never leave.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ezkl  # type: ignore
import numpy as np
import onnx  # type: ignore

from .errors import (
    CircuitCompilationError,
    ModelNotSetupError,
    ProofGenerationError,
    VerificationInputError,
    InputShapeError,
    HINT_DYNAMIC_SHAPES,
    HINT_UNSUPPORTED_OP,
    HINT_NOT_SETUP,
    HINT_INPUT_SHAPE,
)
from .store import (
    Store,
    BundleMeta,
    model_hash_of,
    agent_id_of,
    sha256_file,
    hex0x,
    now_pair,
)


# --------------------------------------------------------------------------
# ONNX inspection
# --------------------------------------------------------------------------
def onnx_input_len(onnx_path: str | os.PathLike) -> int:
    """Flattened length of the model's (single) input tensor.

    Dynamic / unknown dimensions are treated as 1, which is what a fixed ZK
    circuit requires anyway.
    """
    model = onnx.load(str(onnx_path))
    inits = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in inits]
    if not inputs:
        raise CircuitCompilationError(
            "Model has no graph inputs.",
            hint="The ONNX graph exposes no free input tensor. Re-export the "
            "model so its input is a named graph input.",
        )
    tensor = inputs[0]
    dims = tensor.type.tensor_type.shape.dim
    n = 1
    for d in dims:
        v = d.dim_value if (d.dim_value and d.dim_value > 0) else 1
        n *= v
    return int(n)


# --------------------------------------------------------------------------
# SRS (structured reference string) management
# --------------------------------------------------------------------------
def srs_cache_path(store: Store, logrows: int) -> Path:
    return store.root / "srs" / f"kzg_{logrows}.srs"


def ensure_srs(store: Store, logrows: int, dest: Path) -> tuple[Path, str]:
    """Make an SRS for ``logrows`` available at ``dest``; return (path, source).

    The SRS is a *public* parameter shared by prover and verifier. We prefer
    the canonical perpetual-powers-of-tau ceremony (``get_srs``). If the
    ceremony host is unreachable (offline / restricted network), we fall
    back to a locally generated SRS. That fallback is a **trusted setup by
    the issuer** and is flagged loudly; the README threat model explains the
    implication.
    """
    cache = srs_cache_path(store, logrows)
    source = "ceremony"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            asyncio.run(_get_srs_async(logrows, str(cache)))
            source = "ceremony"
        except Exception:
            # Offline / restricted: generate locally (non-canonical).
            ezkl.gen_srs(str(cache), logrows)
            source = "local-generated"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != cache.resolve():
        dest.write_bytes(cache.read_bytes())
    return dest, source


async def _get_srs_async(logrows: int, path: str) -> None:
    await ezkl.get_srs(None, logrows, path)


# --------------------------------------------------------------------------
# Setup: ONNX -> circuit + keys
# --------------------------------------------------------------------------
@dataclass
class SetupResult:
    meta: BundleMeta
    srs_source: str
    setup_time_s: float


def _calibration_input(input_len: int) -> dict[str, Any]:
    # Calibration sizes the circuit's lookup tables and fixed-point scales.
    # A *constant* vector is degenerate and can push EZKL into a pathological
    # circuit that deadlocks the prover, so we use a spread of varied values
    # across a plausible range. For best results, callers should pass a real
    # representative sample via setup(..., calibration_input=...).
    vals = np.linspace(-1.5, 1.5, num=input_len, dtype=np.float64)
    if input_len == 1:
        vals = np.array([0.7])
    return {"input_data": [list(vals)]}


def setup_model(
    onnx_path: str | os.PathLike,
    name: str,
    *,
    task: str = "auto",
    labels: Optional[list[str]] = None,
    threshold: float = 0.5,
    store: Optional[Store] = None,
    calibration_input: Optional[list[float]] = None,
) -> SetupResult:
    """Compile a ZK circuit for an ONNX model and generate its keys.

    One-time cost per model version. Idempotent: re-running with the same
    model and name rebuilds the bundle in place.
    """
    store = store or Store()
    store.ensure_dirs()
    onnx_path = str(onnx_path)

    if not Path(onnx_path).exists():
        raise CircuitCompilationError(f"Model file not found: {onnx_path}")

    model_hash = model_hash_of(onnx_path)
    agent_id = agent_id_of(name, model_hash)
    bundle = store.bundle_dir(agent_id)
    bundle.mkdir(parents=True, exist_ok=True)

    input_len = onnx_input_len(onnx_path)

    # Copy the model into the bundle so the agent_id self-contains it.
    (bundle / "model.onnx").write_bytes(Path(onnx_path).read_bytes())

    settings_path = bundle / "settings.json"
    compiled_path = bundle / "model.compiled"
    pk_path = bundle / "pk.key"
    vk_path = bundle / "vk.key"
    srs_path = bundle / "kzg.srs"

    cal = (
        {"input_data": [list(calibration_input)]}
        if calibration_input is not None
        else _calibration_input(input_len)
    )
    cal_path = bundle / "calibration.json"
    cal_path.write_text(json.dumps(cal))

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "hashed"
    run_args.output_visibility = "public"
    run_args.param_visibility = "fixed"

    t0 = time.time()
    # 1. settings
    try:
        ezkl.gen_settings(onnx_path, str(settings_path), py_run_args=run_args)
    except Exception as e:
        raise _translate_compile_error(e)
    # 2. calibrate (sizes lookup tables / picks scales)
    try:
        ezkl.calibrate_settings(
            str(cal_path), onnx_path, str(settings_path), "resources"
        )
    except Exception as e:
        raise _translate_compile_error(e)
    # 3. compile circuit
    try:
        ezkl.compile_circuit(onnx_path, str(compiled_path), str(settings_path))
    except Exception as e:
        raise _translate_compile_error(e)

    logrows = int(json.loads(settings_path.read_text())["run_args"]["logrows"])

    # 4. SRS
    _, srs_source = ensure_srs(store, logrows, srs_path)

    # 5. setup -> pk / vk
    try:
        ezkl.setup(str(compiled_path), str(vk_path), str(pk_path), str(srs_path))
    except Exception as e:
        raise ProofGenerationError(
            f"Key generation (setup) failed: {e}",
            hint="The circuit compiled but key generation failed. This is "
            "usually an SRS/logrows mismatch. Try deleting the SRS cache "
            "(.zkml-trail/srs) and re-running setup.",
        )
    setup_time = time.time() - t0

    resolved_task = _resolve_task(task, input_len, settings_path)
    output_len = _infer_output_len(settings_path)

    t, human = now_pair()
    meta = BundleMeta(
        agent_id=agent_id,
        name=name,
        model_hash=model_hash,
        vk_hash=hex0x(sha256_file(vk_path)),
        srs_hash=hex0x(sha256_file(srs_path)),
        settings_hash=hex0x(sha256_file(settings_path)),
        logrows=logrows,
        input_len=input_len,
        output_len=output_len,
        created=t,
        created_human=human,
        task=resolved_task,
        labels=labels,
        threshold=threshold,
    )
    store.save_meta(meta)
    store.register_index(meta)
    return SetupResult(meta=meta, srs_source=srs_source, setup_time_s=setup_time)


def _resolve_task(task: str, input_len: int, settings_path: Path) -> str:
    if task != "auto":
        return task
    out_len = _infer_output_len(settings_path)
    if out_len == 1:
        return "binary"
    return "classification"


def _infer_output_len(settings_path: Path) -> int:
    s = json.loads(settings_path.read_text())
    # settings records the model output scales, one per output element.
    scales = s.get("model_output_scales") or []
    if scales:
        return len(scales)
    return 1


# --------------------------------------------------------------------------
# Prove
# --------------------------------------------------------------------------
@dataclass
class ProveResult:
    agent_id: str
    model_hash: str
    output: list[float]
    input_hash: str  # Poseidon hash, bound by the proof
    proof_object: dict[str, Any]
    settings: dict[str, Any]
    proof_hex: str
    proof_size_bytes: int
    vk_hash: str
    srs_hash: str
    logrows: int
    meta: BundleMeta
    witness_time_s: float
    prove_time_s: float


def prove(
    onnx_path: str | os.PathLike,
    input_values: list[float],
    *,
    store: Optional[Store] = None,
    agent_id: Optional[str] = None,
) -> ProveResult:
    """Generate a ZK proof for one inference. The input is never persisted."""
    store = store or Store()
    onnx_path = str(onnx_path)
    model_hash = model_hash_of(onnx_path)

    if agent_id is None:
        agent_id = store.agent_for_model_hash(model_hash)
    if agent_id is None or not store.has_bundle(agent_id):
        raise ModelNotSetupError(
            f"No ZK circuit registered for model {model_hash[:12]}…",
            hint=HINT_NOT_SETUP,
        )

    meta = store.load_meta(agent_id)
    assert meta is not None
    bundle = store.bundle_dir(agent_id)

    flat = list(np.asarray(input_values, dtype=np.float64).reshape(-1))
    if len(flat) != meta.input_len:
        raise InputShapeError(
            f"Input has {len(flat)} values but model expects {meta.input_len}.",
            hint=HINT_INPUT_SHAPE,
        )

    compiled_path = bundle / "model.compiled"
    pk_path = bundle / "pk.key"
    vk_path = bundle / "vk.key"
    srs_path = bundle / "kzg.srs"
    settings_path = bundle / "settings.json"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        input_path = td / "input.json"
        proof_path = td / "proof.json"
        input_path.write_text(json.dumps({"input_data": [flat]}))

        t1 = time.time()
        _run_prove_subprocess(bundle, input_path, proof_path)
        prove_time = time.time() - t1
        witness_time = 0.0  # rolled into the isolated prove step

        proof_object = json.loads(proof_path.read_text())

    output = _decode_outputs(proof_object)
    input_hash = _decode_input_hash(proof_object)
    proof_hex = proof_object.get("hex_proof", "")
    proof_size = len(proof_object.get("proof", [])) or (len(proof_hex) // 2)
    settings = json.loads(settings_path.read_text())

    return ProveResult(
        agent_id=agent_id,
        model_hash=model_hash,
        output=output,
        input_hash=input_hash,
        proof_object=proof_object,
        settings=settings,
        proof_hex=proof_hex,
        proof_size_bytes=proof_size,
        vk_hash=meta.vk_hash,
        srs_hash=meta.srs_hash,
        logrows=meta.logrows,
        meta=meta,
        witness_time_s=witness_time,
        prove_time_s=prove_time,
    )


def _run_prove_subprocess(
    bundle: Path,
    input_path: Path,
    proof_path: Path,
    *,
    timeout_s: int = 90,
    attempts: int = 4,
) -> None:
    """Run witness+prove in an isolated subprocess, with timeout and retry.

    EZKL's native prover can occasionally deadlock (all threads parked on a
    futex) in constrained containers. A deadlock inside a Rust extension is
    not interruptible from Python, so we run it out-of-process and kill +
    retry on timeout. Each retry is a fresh process, which clears the
    wedged thread pool.
    """
    cmd = [
        sys.executable, "-m", "zkml_trail._prove_worker",
        str(bundle), str(input_path), str(proof_path),
    ]
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                cmd, timeout=timeout_s, capture_output=True, text=True
            )
        except subprocess.TimeoutExpired:
            last_err = (
                f"proof attempt {attempt}/{attempts} exceeded {timeout_s}s "
                "(EZKL prover stall) — retrying in a fresh process"
            )
            continue
        if proc.returncode == 0 and proof_path.exists():
            return
        last_err = (proc.stderr or proc.stdout or "unknown error").strip()
        # A non-timeout, non-zero exit is a real error; don't spin on it.
        if proc.returncode != 0 and "panic" not in last_err.lower():
            break
    raise ProofGenerationError(
        f"Proof generation failed after {attempts} attempt(s).",
        hint="The EZKL/Halo2 prover did not complete. Two common causes:\n"
        "  1. The input is far outside the range the circuit was calibrated\n"
        "     for. Re-run setup with a representative --calibration sample.\n"
        "  2. A transient prover stall in constrained environments. Retrying\n"
        "     usually succeeds; give the process more CPU/RAM if it persists.\n"
        f"\nLast error: {last_err}",
    )


def _decode_outputs(proof_object: dict[str, Any]) -> list[float]:
    ppi = proof_object.get("pretty_public_inputs") or {}
    rescaled = ppi.get("rescaled_outputs") or []
    if rescaled:
        return [float(x) for x in rescaled[0]]
    return []


def _decode_input_hash(proof_object: dict[str, Any]) -> str:
    ppi = proof_object.get("pretty_public_inputs") or {}
    processed = ppi.get("processed_inputs") or []
    if processed and processed[0]:
        h = processed[0][0]
        return h if str(h).startswith("0x") else hex0x(str(h))
    return ""


# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
@dataclass
class VerifyResult:
    verified: bool
    agent_id: str
    model_hash: str
    output: list[float]
    decision: str
    input_hash: str
    timestamp: int
    timestamp_human: str
    vk_hash: str
    srs_source: str
    verify_time_s: float
    notes: list[str]


def verify(
    receipt: dict[str, Any],
    vk_path: str | os.PathLike,
    *,
    srs_path: Optional[str | os.PathLike] = None,
    store: Optional[Store] = None,
) -> VerifyResult:
    """Verify a receipt against a public verifying key. Pure math, no model.

    Independent checks performed:
      1. ZK proof is valid for (settings, vk, srs)         -> ezkl.verify
      2. provided vk matches the vk_hash bound in receipt  -> tamper check
      3. provided srs matches the srs_hash bound in receipt-> tamper check
      4. public output in the proof == receipt's stated decision/output
      5. Poseidon input hash in the proof == receipt's input_hash
    """
    store = store or Store()
    zk = receipt.get("_zkml_verification")
    if not zk:
        raise VerificationInputError(
            "Receipt is missing its verification payload (_zkml_verification).",
            hint="This receipt was not produced by zkml-trail >= 1.0, or it "
            "was truncated. A verifiable receipt must embed the proof object "
            "and circuit settings.",
        )

    proof_object = zk["proof"]
    settings = zk["settings"]
    rec_vk_hash = receipt["attestation"]["vk_hash"]
    rec_srs_hash = zk.get("srs_hash", "")
    logrows = int(zk.get("logrows", settings["run_args"]["logrows"]))

    notes: list[str] = []

    vk_path = str(vk_path)
    if not Path(vk_path).exists():
        raise VerificationInputError(f"Verifying key not found: {vk_path}")
    actual_vk_hash = hex0x(sha256_file(vk_path))
    if actual_vk_hash != rec_vk_hash:
        return _failed_verify(
            receipt,
            actual_vk_hash,
            "wrong-vk",
            ["Provided verifying key does not match the vk_hash bound in the "
             "receipt — this is a different key than the one that signed "
             "this decision."],
        )

    # Locate the SRS (public parameter that must accompany the vk).
    resolved_srs, srs_source = _resolve_srs(
        srs_path, vk_path, store, logrows, rec_srs_hash
    )
    if resolved_srs is None:
        raise VerificationInputError(
            "Could not locate the SRS needed to verify this proof.",
            hint="The structured reference string (SRS) is a public "
            "parameter shared by prover and verifier. Place 'kzg.srs' next "
            "to the verifying key, pass --srs <path>, or let zkml-trail find "
            f"it in the local cache for logrows={logrows}.",
        )

    if rec_srs_hash and hex0x(sha256_file(resolved_srs)) != rec_srs_hash:
        return _failed_verify(
            receipt, actual_vk_hash, "wrong-srs",
            ["Provided SRS does not match the srs_hash bound in the receipt."],
        )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pf = td / "proof.json"
        st = td / "settings.json"
        pf.write_text(json.dumps(proof_object))
        st.write_text(json.dumps(settings))
        t0 = time.time()
        try:
            ok = bool(ezkl.verify(str(pf), str(st), vk_path, str(resolved_srs)))
        except Exception as e:
            return _failed_verify(
                receipt, actual_vk_hash, "proof-invalid",
                [f"EZKL rejected the proof: {e}"],
            )
        verify_time = time.time() - t0

    # Cross-check: the public values inside the proof must match the receipt.
    proof_output = _decode_outputs(proof_object)
    proof_input_hash = _decode_input_hash(proof_object)
    if _round(proof_output) != _round(receipt["decision"]["output"]):
        ok = False
        notes.append(
            "Receipt's stated output does not match the public output bound "
            "in the proof — the human-readable decision was altered."
        )
    if proof_input_hash and proof_input_hash != receipt["decision"]["input_hash"]:
        ok = False
        notes.append(
            "Receipt's input_hash does not match the Poseidon commitment "
            "bound in the proof."
        )

    return VerifyResult(
        verified=ok,
        agent_id=receipt["agent"]["agent_id"],
        model_hash=receipt["agent"]["model_hash"],
        output=proof_output,
        decision=receipt["decision"]["decision"],
        input_hash=receipt["decision"]["input_hash"],
        timestamp=receipt["zkml_trail"]["timestamp"],
        timestamp_human=receipt["zkml_trail"]["timestamp_human"],
        vk_hash=actual_vk_hash,
        srs_source=srs_source,
        verify_time_s=verify_time,
        notes=notes,
    )


def _resolve_srs(srs_path, vk_path, store, logrows, rec_srs_hash):
    candidates = []
    if srs_path:
        candidates.append(Path(srs_path))
    candidates.append(Path(vk_path).parent / "kzg.srs")
    candidates.append(srs_cache_path(store, logrows))
    for c in candidates:
        if c and c.exists():
            return c, "provided" if srs_path else "local-cache"
    return None, ""


def _failed_verify(receipt, vk_hash, source, notes) -> VerifyResult:
    return VerifyResult(
        verified=False,
        agent_id=receipt.get("agent", {}).get("agent_id", "unknown"),
        model_hash=receipt.get("agent", {}).get("model_hash", "unknown"),
        output=receipt.get("decision", {}).get("output", []),
        decision=receipt.get("decision", {}).get("decision", "unknown"),
        input_hash=receipt.get("decision", {}).get("input_hash", ""),
        timestamp=receipt.get("zkml_trail", {}).get("timestamp", 0),
        timestamp_human=receipt.get("zkml_trail", {}).get("timestamp_human", ""),
        vk_hash=vk_hash,
        srs_source=source,
        verify_time_s=0.0,
        notes=notes,
    )


def _round(xs, ndigits: int = 5):
    try:
        return [round(float(x), ndigits) for x in xs]
    except Exception:
        return xs


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------
def _translate_compile_error(e: Exception) -> CircuitCompilationError:
    msg = str(e).lower()
    if "dynamic" in msg or "dim_param" in msg or "unknown dimension" in msg:
        return CircuitCompilationError(
            "Circuit compilation failed: model has dynamic shapes.",
            hint=HINT_DYNAMIC_SHAPES,
        )
    if "unsupported" in msg or "not supported" in msg or "no support" in msg:
        return CircuitCompilationError(
            "Circuit compilation failed: unsupported operator.",
            hint=HINT_UNSUPPORTED_OP,
        )
    return CircuitCompilationError(
        f"Circuit compilation failed: {e}",
        hint="EZKL could not lower this ONNX graph into a Halo2 circuit. "
        "Run onnx-simplifier on the model, confirm all shapes are static, "
        "and check the operators against EZKL's supported-ops list.",
    )
