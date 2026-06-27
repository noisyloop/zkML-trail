"""zkml-trail command-line interface.

Design principles baked in here:
  * Every command prints performance numbers (time + proof size) so users
    can decide whether the ZK overhead is worth it for their use case.
  * Errors explain the ZK concept that failed and how to fix it.
  * verify needs only the receipt and the public verifying key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .version import __version__
from .store import Store
from .errors import ZkmlTrailError
from .backend import setup_model, prove as backend_prove, verify as backend_verify
from .receipt import build_receipt, inspect_receipt, decode_decision
from .inputs import load_input_file


# --- pretty printing helpers ---------------------------------------------
def _err(msg: str) -> None:
    click.secho(msg, fg="red", err=True)


def _ok(msg: str) -> None:
    click.secho(msg, fg="green")


def _hl(msg: str) -> None:
    click.secho(msg, fg="cyan")


def _perf(label: str, value: str) -> None:
    click.secho(f"  ⏱  {label}: ", nl=False, fg="yellow")
    click.echo(value)


def _handle(e: ZkmlTrailError) -> None:
    _err(f"\n✗ {e.message}")
    if e.hint:
        click.secho("\n  ┌─ what this means ──────────────────────────────",
                    fg="white", err=True)
        for line in e.hint.splitlines():
            click.secho(f"  │ {line}", fg="white", err=True)
        click.secho("  └────────────────────────────────────────────────",
                    fg="white", err=True)
    sys.exit(1)


@click.group()
@click.version_option(__version__, prog_name="zkml-trail")
def cli() -> None:
    """Cryptographic receipts for AI agent decisions (ZK proofs via EZKL)."""


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
@cli.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", required=True, help="Human name for this agent/model.")
@click.option("--task", default="auto",
              type=click.Choice(["auto", "classification", "binary", "score", "raw"]),
              help="How to decode the output vector into a decision.")
@click.option("--labels", default=None,
              help="Comma-separated class labels aligned to output index.")
@click.option("--threshold", default=0.5, type=float,
              help="Decision threshold for --task binary.")
@click.option("--calibration", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Optional representative input JSON for "
                                 "circuit calibration.")
def setup(model, name, task, labels, threshold, calibration):
    """Compile a ZK circuit from MODEL (ONNX) and generate keys. One-time.

    Produces a proving key (private, the agent keeps) and a verifying key
    (public, share freely), then prints the model_id.
    """
    label_list = [s.strip() for s in labels.split(",")] if labels else None
    cal = None
    if calibration:
        cal = load_input_file(calibration)
    store = Store()
    _hl(f"\nCompiling ZK circuit for '{name}' …")
    click.echo("  (one-time cost per model version; proving is cheap after)")
    try:
        res = setup_model(
            model, name, task=task, labels=label_list, threshold=threshold,
            store=store, calibration_input=cal,
        )
    except ZkmlTrailError as e:
        _handle(e)
        return
    m = res.meta
    bundle = store.bundle_dir(m.agent_id)
    _ok("\n✓ Circuit compiled and keys generated.")
    click.echo("")
    _hl("  model_id (model hash, share this):")
    click.echo(f"    {m.model_hash}")
    click.echo("")
    click.echo(f"  agent_id        : {m.agent_id}")
    click.echo(f"  task            : {m.task}" + (
        f"  labels={m.labels}" if m.labels else ""))
    click.echo(f"  input length    : {m.input_len}")
    click.echo(f"  circuit logrows : {m.logrows}  (2^{m.logrows} rows)")
    click.echo("")
    click.echo("  Artifacts:")
    click.echo(f"    proving key  (PRIVATE): {bundle/'pk.key'}")
    click.echo(f"    verifying key (PUBLIC): {bundle/'vk.key'}")
    click.echo(f"    settings.json         : {bundle/'settings.json'}")
    click.echo(f"    SRS  ({res.srs_source:<16}): {bundle/'kzg.srs'}")
    if res.srs_source == "local-generated":
        click.secho(
            "\n  ⚠  SRS was generated locally (ceremony host unreachable).\n"
            "     This is a *trusted setup by the issuer*. For production,\n"
            "     use the canonical perpetual-powers-of-tau SRS so verifiers\n"
            "     need not trust you. See README › threat model.",
            fg="yellow")
    click.echo("")
    _perf("setup time", f"{res.setup_time_s:.2f}s")
    click.echo("")
    _hl("  Next: zkml-trail prove --model <model.onnx> --input <input.json> "
        "--output receipt.json")


# --------------------------------------------------------------------------
# prove
# --------------------------------------------------------------------------
@cli.command()
@click.option("--model", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", default="receipt.json",
              help="Where to write the receipt.")
def prove(model, input_path, output_path):
    """Run inference privately and emit a verifiable receipt.

    The input is never stored; the model weights are never exposed.
    """
    store = Store()
    try:
        values = load_input_file(input_path)
    except Exception as e:
        _err(f"✗ Could not read input '{input_path}': {e}")
        sys.exit(1)

    _hl("\nGenerating zero-knowledge proof …")
    click.echo("  running model inference privately (input stays local)")
    try:
        res = backend_prove(model, values, store=store)
    except ZkmlTrailError as e:
        _handle(e)
        return

    receipt = build_receipt(res)
    Path(output_path).write_text(json.dumps(receipt, indent=2))
    store.save_receipt(res.agent_id, receipt)

    decision = receipt["decision"]["decision"]
    conf = receipt["decision"]["confidence"]
    _ok(f"\n✓ Receipt issued: {receipt['zkml_trail']['receipt_id']}")
    click.echo("")
    click.echo(f"  decision   : {decision}" + (
        f"  (confidence {conf})" if conf is not None else ""))
    click.echo(f"  output     : {receipt['decision']['output']}")
    click.echo(f"  input hash : {res.input_hash[:18]}…  (bound by the proof)")
    click.echo(f"  model hash : {res.model_hash}")
    click.echo(f"  written to : {output_path}")
    click.echo("")
    _perf("witness time", f"{res.witness_time_s:.2f}s")
    _perf("proof time  ", f"{res.prove_time_s:.2f}s")
    _perf("proof size  ", f"{res.proof_size_bytes} bytes")
    click.echo("")
    _hl("  Anyone can verify: zkml-trail verify "
        f"{output_path} --vk <vk.key>")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
@cli.command()
@click.argument("receipt", type=click.Path(exists=True, dir_okay=False))
@click.option("--vk", "vk_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Public verifying key (vk.key).")
@click.option("--srs", "srs_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="SRS file (defaults to one beside the vk, then cache).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON only.")
def verify(receipt, vk_path, srs_path, as_json):
    """Verify a RECEIPT mathematically. No model, no input — just math."""
    store = Store()
    data = json.loads(Path(receipt).read_text())
    try:
        res = backend_verify(data, vk_path, srs_path=srs_path, store=store)
    except ZkmlTrailError as e:
        _handle(e)
        return

    if as_json:
        click.echo(json.dumps({
            "verified": res.verified,
            "agent_id": res.agent_id,
            "model_id": res.model_hash,
            "output": res.output,
            "decision": res.decision,
            "input": "[private — not stored]",
            "input_hash": res.input_hash,
            "timestamp": res.timestamp,
            "timestamp_human": res.timestamp_human,
            "vk_hash": res.vk_hash,
            "notes": res.notes,
        }, indent=2))
        sys.exit(0 if res.verified else 2)

    if res.verified:
        _ok("\n✓ VERIFIED — this receipt is mathematically valid.\n")
    else:
        _err("\n✗ NOT VERIFIED — do not trust this receipt.\n")
    click.echo(f"  verified   : {res.verified}")
    click.echo(f"  model_id   : {res.model_hash}")
    click.echo(f"  agent_id   : {res.agent_id}")
    click.echo(f"  output     : {res.output}")
    click.echo(f"  decision   : {res.decision}")
    click.echo(f"  input      : [private — not revealed by this proof]")
    click.echo(f"  input_hash : {res.input_hash}")
    click.echo(f"  timestamp  : {res.timestamp_human}")
    click.echo(f"  vk_hash    : {res.vk_hash}")
    click.echo(f"  srs source : {res.srs_source}")
    for n in res.notes:
        click.secho(f"  ! {n}", fg="red")
    click.echo("")
    _perf("verify time", f"{res.verify_time_s*1000:.0f}ms")
    sys.exit(0 if res.verified else 2)


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------
@cli.command()
@click.argument("receipt", type=click.Path(exists=True, dir_okay=False))
def inspect(receipt):
    """Human-readable breakdown: what's proven, what's private, honest caveats."""
    data = json.loads(Path(receipt).read_text())
    click.echo(inspect_receipt(data))


# --------------------------------------------------------------------------
# agents (local registry helper)
# --------------------------------------------------------------------------
@cli.command()
def agents():
    """List models/agents registered in the local artifact store."""
    store = Store()
    reg = store.list_agents()
    if not reg:
        click.echo("No agents registered yet. Run: zkml-trail setup <model.onnx> "
                   "--name <name>")
        return
    for aid, info in reg.items():
        click.echo(f"  {aid}  {info['name']:<24} {info['model_hash'][:14]}…")


# --------------------------------------------------------------------------
# serve (run the API)
# --------------------------------------------------------------------------
@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    """Run the FastAPI proof/verification server."""
    import uvicorn
    uvicorn.run("zkml_trail.api:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    cli()
