"""Subprocess worker that runs witness generation + proving in isolation.

Why a subprocess? EZKL's Halo2 prover (a Rust extension) can, under some
circuit/threading conditions in constrained containers, deadlock with all
worker threads parked on a futex. Because that happens *inside* a native
extension, it cannot be interrupted from Python. Running it in a separate
process lets the parent enforce a wall-clock timeout (kill + retry) instead
of hanging the whole CLI / API.

Invoked as:
    python -m zkml_trail._prove_worker <bundle_dir> <input_json> <out_proof>

``input_json`` is an EZKL-format file ({"input_data": [[...]]}). It is a
caller-managed temp file and is never persisted by zkml-trail.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import ezkl  # type: ignore


def main(argv: list[str]) -> int:
    bundle = Path(argv[0])
    input_json = argv[1]
    out_proof = argv[2]

    compiled = str(bundle / "model.compiled")
    pk = str(bundle / "pk.key")
    srs = str(bundle / "kzg.srs")

    with tempfile.TemporaryDirectory() as td:
        witness = str(Path(td) / "witness.json")
        # gen_witness without vk/srs keeps the witness in the plain form the
        # standalone prover expects (no extra KZG commitments).
        ezkl.gen_witness(input_json, compiled, witness)
        ezkl.prove(witness, compiled, pk, out_proof, srs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
