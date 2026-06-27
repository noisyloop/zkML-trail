"""Artifact store: where per-model ZK material lives on disk.

zkml-trail keeps one *bundle* per registered model version. A bundle holds
the compiled circuit, the proving key (private), the verifying key
(public), the SRS (public structured reference string) and metadata. The
store also records an index mapping model hashes / names / agent ids to
bundles, plus issued receipts (for API history).

Layout (under ZKML_TRAIL_HOME, default ./.zkml-trail):

    .zkml-trail/
      index.json                     # name/hash/agent -> agent_id
      agents/<agent_id>/
        meta.json                    # name, hashes, shapes, decode config
        model.onnx                   # copy of the registered model
        settings.json                # EZKL circuit settings (public)
        model.compiled               # compiled circuit
        pk.key                       # proving key  (PRIVATE - agent keeps)
        vk.key                       # verifying key (PUBLIC - share freely)
        kzg.srs                      # structured reference string (public)
      receipts/<receipt_id>.json     # issued receipts (API history)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


def home_dir() -> Path:
    """Root of the local artifact store.

    Honors ``ZKML_TRAIL_HOME``; otherwise uses ``./.zkml-trail`` so a repo
    checkout is self-contained and predictable.
    """
    return Path(os.environ.get("ZKML_TRAIL_HOME", ".zkml-trail")).resolve()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hex0x(hexdigest: str) -> str:
    return hexdigest if hexdigest.startswith("0x") else "0x" + hexdigest


def model_hash_of(onnx_path: str | os.PathLike) -> str:
    """Stable identifier for a model *version*: 0x + sha256 of the file.

    Two byte-identical ONNX files (same graph, same weights) get the same
    hash; changing a single weight changes it. This is the value a verifier
    checks to confirm "the correct model version ran".
    """
    return hex0x(sha256_file(onnx_path))


def agent_id_of(name: str, model_hash: str) -> str:
    """Short, human-friendly agent id derived from name + model hash."""
    digest = sha256_bytes(f"{name}::{model_hash}".encode())[:12]
    return f"agent_{digest}"


@dataclass
class BundleMeta:
    agent_id: str
    name: str
    model_hash: str
    vk_hash: str
    srs_hash: str
    settings_hash: str
    logrows: int
    input_len: int
    output_len: int
    created: int
    created_human: str
    # Decode configuration: how to turn a raw output vector into a decision.
    task: str  # "classification" | "binary" | "score" | "raw"
    labels: Optional[list[str]] = None  # class names, aligned to output index
    threshold: float = 0.5  # for "binary"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class Store:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or home_dir()
        self.agents_dir = self.root / "agents"
        self.receipts_dir = self.root / "receipts"
        self.index_path = self.root / "index.json"

    # --- bundle paths -----------------------------------------------------
    def bundle_dir(self, agent_id: str) -> Path:
        return self.agents_dir / agent_id

    def ensure_dirs(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    # --- index ------------------------------------------------------------
    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"by_model_hash": {}, "by_name": {}, "agents": {}}

    def _save_index(self, idx: dict[str, Any]) -> None:
        self.ensure_dirs()
        self.index_path.write_text(json.dumps(idx, indent=2))

    def register_index(self, meta: BundleMeta) -> None:
        idx = self._load_index()
        idx["by_model_hash"][meta.model_hash] = meta.agent_id
        idx["by_name"][meta.name] = meta.agent_id
        idx["agents"][meta.agent_id] = {
            "name": meta.name,
            "model_hash": meta.model_hash,
            "created": meta.created,
        }
        self._save_index(idx)

    def agent_for_model_hash(self, model_hash: str) -> Optional[str]:
        return self._load_index()["by_model_hash"].get(model_hash)

    def agent_for_name(self, name: str) -> Optional[str]:
        return self._load_index()["by_name"].get(name)

    def list_agents(self) -> dict[str, Any]:
        return self._load_index()["agents"]

    # --- meta -------------------------------------------------------------
    def save_meta(self, meta: BundleMeta) -> None:
        d = self.bundle_dir(meta.agent_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta.to_json(), indent=2))

    def load_meta(self, agent_id: str) -> Optional[BundleMeta]:
        p = self.bundle_dir(agent_id) / "meta.json"
        if not p.exists():
            return None
        return BundleMeta(**json.loads(p.read_text()))

    def has_bundle(self, agent_id: str) -> bool:
        d = self.bundle_dir(agent_id)
        return (d / "pk.key").exists() and (d / "model.compiled").exists()

    # --- receipts (API history) ------------------------------------------
    def save_receipt(self, agent_id: str, receipt: dict[str, Any]) -> None:
        self.ensure_dirs()
        rid = receipt["zkml_trail"]["receipt_id"]
        (self.receipts_dir / f"{rid}.json").write_text(json.dumps(receipt, indent=2))
        # append to per-agent history index
        hist_path = self.bundle_dir(agent_id) / "history.json"
        hist = []
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
        hist.append(
            {
                "receipt_id": rid,
                "timestamp": receipt["zkml_trail"]["timestamp"],
                "decision": receipt["decision"]["decision"],
                "output": receipt["decision"]["output"],
            }
        )
        hist_path.write_text(json.dumps(hist, indent=2))

    def load_receipt(self, receipt_id: str) -> Optional[dict[str, Any]]:
        p = self.receipts_dir / f"{receipt_id}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def agent_history(self, agent_id: str) -> list[dict[str, Any]]:
        p = self.bundle_dir(agent_id) / "history.json"
        if not p.exists():
            return []
        return json.loads(p.read_text())


def now_pair() -> tuple[int, str]:
    """Return (unix_seconds, ISO-8601 UTC string)."""
    t = int(time.time())
    human = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    return t, human
