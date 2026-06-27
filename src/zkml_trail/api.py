"""FastAPI server: register agents, issue receipts, verify, serve history.

This is the networked face of the same engine the CLI uses. It deliberately
mirrors the CLI guarantees:
  * decision inputs are never stored,
  * verifying keys are public and downloadable,
  * every receipt is independently verifiable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .version import __version__
from .store import Store
from .errors import ZkmlTrailError
from .backend import setup_model, prove as backend_prove, verify as backend_verify
from .receipt import build_receipt
from .inputs import parse_input


app = FastAPI(
    title="zkml-trail",
    version=__version__,
    description="Cryptographic receipts for AI agent decisions (ZK proofs via "
    "EZKL / Halo2).",
)

_store = Store()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "zkml-trail",
        "version": __version__,
        "proving_system": "Halo2/EZKL",
        "endpoints": [
            "POST /agents/register",
            "POST /agents/{agent_id}/decide",
            "POST /receipts/verify",
            "GET  /receipts/{receipt_id}",
            "GET  /agents/{agent_id}/vk",
            "GET  /agents/{agent_id}/history",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------
@app.post("/agents/register")
async def register_agent(
    model: UploadFile = File(..., description="ONNX model file"),
    name: str = Form(...),
    task: str = Form("auto"),
    labels: Optional[str] = Form(None),
    threshold: float = Form(0.5),
) -> dict[str, Any]:
    """Upload an ONNX model and compile its ZK circuit. One-time per version."""
    label_list = [s.strip() for s in labels.split(",")] if labels else None
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        tf.write(await model.read())
        tmp_path = tf.name
    try:
        res = setup_model(
            tmp_path, name, task=task, labels=label_list,
            threshold=threshold, store=_store,
        )
    except ZkmlTrailError as e:
        raise HTTPException(status_code=400, detail={"error": e.message, "hint": e.hint})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    m = res.meta
    return {
        "agent_id": m.agent_id,
        "model_hash": m.model_hash,
        "vk_key_url": f"/agents/{m.agent_id}/vk",
        "task": m.task,
        "labels": m.labels,
        "input_len": m.input_len,
        "logrows": m.logrows,
        "srs_source": res.srs_source,
        "setup_time_s": round(res.setup_time_s, 3),
    }


# --------------------------------------------------------------------------
# Decide (prove)
# --------------------------------------------------------------------------
class DecideRequest(BaseModel):
    decision_input: Any


@app.post("/agents/{agent_id}/decide")
def decide(agent_id: str, req: DecideRequest) -> dict[str, Any]:
    """Run inference privately for AGENT and return a receipt. Input not stored."""
    meta = _store.load_meta(agent_id)
    if meta is None or not _store.has_bundle(agent_id):
        raise HTTPException(status_code=404, detail=f"Unknown agent {agent_id}")

    try:
        values = parse_input(req.decision_input)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bad decision_input: {e}")

    onnx_path = _store.bundle_dir(agent_id) / "model.onnx"
    try:
        res = backend_prove(str(onnx_path), values, store=_store, agent_id=agent_id)
    except ZkmlTrailError as e:
        raise HTTPException(status_code=400, detail={"error": e.message, "hint": e.hint})

    receipt = build_receipt(res)
    _store.save_receipt(agent_id, receipt)
    return {
        "receipt_id": receipt["zkml_trail"]["receipt_id"],
        "decision_output": {
            "output": receipt["decision"]["output"],
            "decision": receipt["decision"]["decision"],
            "confidence": receipt["decision"]["confidence"],
        },
        "performance": {
            "witness_time_s": round(res.witness_time_s, 3),
            "prove_time_s": round(res.prove_time_s, 3),
            "proof_size_bytes": res.proof_size_bytes,
        },
        "receipt": receipt,
    }


# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    receipt: dict[str, Any]
    vk_key: Optional[str] = None  # PEM/base64 vk contents; optional


@app.post("/receipts/verify")
def verify_receipt(req: VerifyRequest) -> dict[str, Any]:
    """Verify a receipt. The vk can be supplied inline or resolved from store."""
    receipt = req.receipt
    agent_id = receipt.get("agent", {}).get("agent_id")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Resolve the verifying key: inline contents win, else the store copy.
        if req.vk_key:
            vk_path = td / "vk.key"
            vk_path.write_text(req.vk_key)
        elif agent_id and _store.has_bundle(agent_id):
            vk_path = _store.bundle_dir(agent_id) / "vk.key"
        else:
            raise HTTPException(
                status_code=400,
                detail="No verifying key: pass vk_key or use a known agent_id.",
            )
        try:
            res = backend_verify(receipt, str(vk_path), store=_store)
        except ZkmlTrailError as e:
            raise HTTPException(status_code=400, detail={"error": e.message, "hint": e.hint})

    return {
        "verified": res.verified,
        "agent_id": res.agent_id,
        "model_hash": res.model_hash,
        "decision": res.decision,
        "output": res.output,
        "input": "[private — not stored]",
        "input_hash": res.input_hash,
        "timestamp": res.timestamp,
        "timestamp_human": res.timestamp_human,
        "vk_hash": res.vk_hash,
        "notes": res.notes,
    }


# --------------------------------------------------------------------------
# Receipt lookup
# --------------------------------------------------------------------------
@app.get("/receipts/{receipt_id}")
def get_receipt(receipt_id: str) -> dict[str, Any]:
    """Public receipt data: decision visible, proof included, input redacted."""
    receipt = _store.load_receipt(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail=f"Unknown receipt {receipt_id}")
    # Input is already redacted in stored receipts ("[private — ZK proven]").
    return receipt


# --------------------------------------------------------------------------
# Verifying key download
# --------------------------------------------------------------------------
@app.get("/agents/{agent_id}/vk")
def get_vk(agent_id: str):
    """Download the public verifying key for independent verification."""
    p = _store.bundle_dir(agent_id) / "vk.key"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Unknown agent {agent_id}")
    return FileResponse(str(p), media_type="application/octet-stream",
                        filename=f"{agent_id}_vk.key")


@app.get("/agents/{agent_id}/srs")
def get_srs_file(agent_id: str):
    """Download the public SRS (needed alongside the vk to verify)."""
    p = _store.bundle_dir(agent_id) / "kzg.srs"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Unknown agent {agent_id}")
    return FileResponse(str(p), media_type="application/octet-stream",
                        filename=f"{agent_id}_kzg.srs")


# --------------------------------------------------------------------------
# History (audit trail)
# --------------------------------------------------------------------------
@app.get("/agents/{agent_id}/history")
def get_history(agent_id: str) -> dict[str, Any]:
    """Full audit trail: receipt ids, timestamps, decisions. Inputs redacted."""
    meta = _store.load_meta(agent_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent {agent_id}")
    hist = _store.agent_history(agent_id)
    return {
        "agent_id": agent_id,
        "model_hash": meta.model_hash,
        "count": len(hist),
        "receipts": [
            {
                "receipt_id": h["receipt_id"],
                "timestamp": h["timestamp"],
                "decision": h["decision"],
                "output": h["output"],
                "input": "[private — never stored]",
            }
            for h in hist
        ],
    }
