"""Receipt construction, decision decoding, and human-readable inspection."""

from __future__ import annotations

import math
import secrets
from typing import Any, Optional

from .version import RECEIPT_VERSION, PROVING_SYSTEM
from .store import BundleMeta, now_pair, sha256_bytes, hex0x
from .backend import ProveResult


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def decode_decision(
    output: list[float], meta: BundleMeta
) -> tuple[str, Optional[float]]:
    """Turn a raw output vector into a (decision, confidence) pair.

    The decoding is configured at setup time (task + labels). It is purely
    presentational: the cryptographic content of the receipt is the raw
    ``output`` vector bound by the proof. ``decision``/``confidence`` are a
    convenience label on top of it.
    """
    task = meta.task
    labels = meta.labels

    if not output:
        return "unknown", None

    if task == "raw":
        return "raw", None

    if task == "score":
        score = float(output[0])
        return f"{round(score, 4)}", None

    if task == "binary":
        p = float(output[0])
        pos = labels[1] if labels and len(labels) > 1 else "approved"
        neg = labels[0] if labels and len(labels) > 0 else "denied"
        if p >= meta.threshold:
            return pos, round(p, 4)
        return neg, round(1.0 - p, 4)

    # classification (default)
    probs = output
    if any(v < 0.0 or v > 1.0 for v in output) or abs(sum(output) - 1.0) > 0.25:
        probs = _softmax(output)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
    return label, round(float(probs[idx]), 4)


def build_receipt(result: ProveResult) -> dict[str, Any]:
    """Assemble a full receipt dict (spec schema + embedded verification)."""
    meta = result.meta
    t, human = now_pair()
    receipt_id = "rcpt_" + secrets.token_hex(4)

    decision, confidence = decode_decision(result.output, meta)
    output = [round(float(x), 6) for x in result.output]

    vk_hash_short = result.vk_hash[:10] if result.vk_hash else ""
    proof_preview = (
        result.proof_hex
        if len(result.proof_hex) <= 18
        else result.proof_hex[:10] + "…" + result.proof_hex[-6:]
    )

    receipt: dict[str, Any] = {
        "zkml_trail": {
            "version": RECEIPT_VERSION,
            "receipt_id": receipt_id,
            "timestamp": t,
            "timestamp_human": human,
        },
        "agent": {
            "agent_id": meta.agent_id,
            "model_hash": meta.model_hash,
            "model_version": "[private]",
            "model_weights": "[private — ZK proven]",
        },
        "decision": {
            "input": "[private — ZK proven]",
            "input_hash": result.input_hash,
            "output": output,
            "decision": decision,
            "confidence": confidence,
        },
        "attestation": {
            "proof": proof_preview,
            "proving_system": PROVING_SYSTEM,
            "vk_hash": result.vk_hash,
            "vk_hash_short": vk_hash_short,
            "verified": True,
            "proof_size_bytes": result.proof_size_bytes,
            "logrows": result.logrows,
        },
        "guarantees": {
            "model_ran_correctly": True,
            "input_was_valid": True,
            "output_not_tampered": True,
            "human_did_not_override": True,
            "correct_model_version": True,
        },
        "limitations": {
            "input_quality": "not verified — garbage in, garbage out",
            "model_bias": "not verified — ZK proves execution, not fairness",
            "network_privacy": "not verified — use separately",
            "permissions": "not verified — ZK does not check authorization",
        },
        "public_statement": (
            f"Agent {meta.agent_id} ran model {meta.model_hash[:10]} on a "
            f"private input (Poseidon hash {result.input_hash[:10]}) and "
            f"produced decision '{decision}'. This is mathematically "
            f"verified by a Halo2/EZKL zero-knowledge proof. The input and "
            f"model weights remain private."
        ),
        # Non-spec, machine-only block: everything an independent verifier
        # needs. Kept separate from the human-facing schema above.
        "_zkml_verification": {
            "proof": result.proof_object,
            "settings": result.settings,
            "logrows": result.logrows,
            "srs_hash": result.srs_hash,
            "vk_hash": result.vk_hash,
            "proof_full_hex": result.proof_hex,
        },
    }
    return receipt


# --------------------------------------------------------------------------
# Inspect (human-readable breakdown)
# --------------------------------------------------------------------------
def inspect_receipt(receipt: dict[str, Any]) -> str:
    """Render a plain-text, honest breakdown of a receipt."""
    z = receipt.get("zkml_trail", {})
    a = receipt.get("agent", {})
    d = receipt.get("decision", {})
    att = receipt.get("attestation", {})
    g = receipt.get("guarantees", {})
    lim = receipt.get("limitations", {})

    lines: list[str] = []
    add = lines.append

    add("══════════════════════════════════════════════════════════════")
    add("  zkml-trail receipt — human-readable breakdown")
    add("══════════════════════════════════════════════════════════════")
    add("")
    add(f"  Receipt ID : {z.get('receipt_id')}")
    add(f"  Issued     : {z.get('timestamp_human')}  ({z.get('timestamp')})")
    add(f"  Schema     : zkml-trail v{z.get('version')}")
    add("")
    add("  ── WHAT IS PUBLIC (anyone can see) ──────────────────────────")
    add(f"    Agent id        : {a.get('agent_id')}")
    add(f"    Model hash      : {a.get('model_hash')}")
    add(f"    Decision        : {d.get('decision')}")
    conf = d.get("confidence")
    add(f"    Confidence      : {conf if conf is not None else 'n/a'}")
    add(f"    Raw output      : {d.get('output')}")
    add(f"    Input hash      : {d.get('input_hash')}  (Poseidon, in-circuit)")
    add(f"    Proving system  : {att.get('proving_system')}")
    add(f"    Verifying key   : {att.get('vk_hash')}")
    add(f"    Proof size      : {att.get('proof_size_bytes')} bytes")
    add("")
    add("  ── WHAT IS PRIVATE (redacted, never revealed) ───────────────")
    add(f"    Input data      : {d.get('input')}")
    add(f"    Model weights   : {a.get('model_weights')}")
    add(f"    Model version   : {a.get('model_version')}")
    add("")
    add("  ── WHAT THIS RECEIPT GUARANTEES ─────────────────────────────")
    for k, v in g.items():
        mark = "✓" if v else "✗"
        add(f"    {mark} {k.replace('_', ' ')}")
    add("")
    add("  ── WHAT THIS RECEIPT DOES *NOT* GUARANTEE (honest) ──────────")
    for k, v in lim.items():
        add(f"    ✗ {k.replace('_', ' ')}: {v}")
    add("")
    add("  ── PLAIN-LANGUAGE STATEMENT ─────────────────────────────────")
    stmt = receipt.get("public_statement", "")
    for chunk in _wrap(stmt, 60):
        add(f"    {chunk}")
    add("")
    add("  To verify independently:")
    add(f"    zkml-trail verify <this-file> --vk vk.key")
    add("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out
