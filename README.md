# zkml-trail

**Cryptographic receipts for AI agent decisions.**

Every AI agent that takes a consequential action needs a tamper-evident,
verifiable record that:

- the **AI** actually made the decision (not a human),
- the **correct model version** ran (not a different one),
- the **input was real** (not fabricated),
- the **output was not tampered with**,
- …**without revealing the input or the model weights**.

`zkml-trail` generates those receipts using zero-knowledge proofs
(**Halo2** via **EZKL**). A receipt is a small JSON file that anyone can
verify, forever, with nothing but a public verifying key — no model, no
input, no trust in the issuer.

```
Agent runs  →  ZK proof generated  →  receipt issued
Receipt proves: model ran, input was valid, output is authentic
Anyone verifies: mathematically, independently, without seeing private data
```

---

## Quickstart (first verified receipt in ~5 minutes)

```bash
pip install -e .            # installs the `zkml-trail` CLI

# 1. one-time per model version: compile a ZK circuit + keys
zkml-trail setup model.onnx --name "trading-agent-v2" \
    --task classification --labels "sell,hold,buy" \
    --calibration sample_input.json

# 2. per decision: run inference privately, emit a receipt
zkml-trail prove --model model.onnx --input decision_input.json \
    --output receipt.json

# 3. anyone, anytime: verify with only the public verifying key
zkml-trail verify receipt.json --vk .zkml-trail/agents/<agent_id>/vk.key

# human-readable breakdown (what's proven, what's private, honest caveats)
zkml-trail inspect receipt.json
```

Or run a complete example end-to-end:

```bash
python scripts/make_example_models.py     # build the tiny demo models
examples/trading-agent/verify.sh          # setup → prove → verify, one command
```

---

## What a receipt looks like

```json
{
  "zkml_trail": { "version": "1.0.0", "receipt_id": "rcpt_7f3a9b2c", "timestamp": 1753948740 },
  "agent":   { "agent_id": "agent_4d5e6f", "model_hash": "0x4d5e6f…",
               "model_weights": "[private — ZK proven]" },
  "decision":{ "input": "[private — ZK proven]", "input_hash": "0x9b2c3d…",
               "output": [0.94, 0.06], "decision": "approved", "confidence": 0.94 },
  "attestation": { "proof": "0x1a2b…", "proving_system": "Halo2/EZKL",
                   "vk_hash": "0x7f3a…", "verified": true, "proof_size_bytes": 4416 },
  "guarantees":  { "model_ran_correctly": true, "human_did_not_override": true, … },
  "limitations": { "model_bias": "not verified — ZK proves execution, not fairness", … },
  "public_statement": "Agent agent_4d5e6f ran model 0x4d5e6f on a private input and produced this decision. This is mathematically verified. The input and model remain private."
}
```

The receipt also carries a machine-only `_zkml_verification` block (the full
proof object + circuit settings) so it is **self-contained**: a verifier
needs only the receipt, the public verifying key, and the public SRS.

---

## What zkml-trail proves — and what it does NOT

This section is **required reading** and intentionally blunt. zkml-trail
should never overclaim.

**zkml-trail proves:**

- ✅ the *specified* model (a fixed set of weights, identified by hash) ran
- ✅ on an input whose Poseidon hash is bound into the proof
- ✅ producing *this specific* output
- ✅ and nothing was tampered with after the fact

**zkml-trail does NOT prove:**

- ❌ the input was accurate or honest (*garbage in, garbage out*)
- ❌ the model is fair or unbiased (ZK proves *execution*, not *fairness*)
- ❌ the model is the *right* model for this decision
- ❌ the decision was ethical or correct
- ❌ network-level privacy (use a separate transport layer)
- ❌ the agent had the correct permissions/authorization

A receipt is a proof of **faithful execution**, not a proof of good
judgment. Treat it accordingly.

---

## Honest comparison: zkml-trail vs. just signing the output

| | Signing the output (e.g. Ed25519) | zkml-trail (ZK proof) |
|---|---|---|
| Speed | microseconds | seconds (per proof) |
| Size | ~64 bytes | a few KB |
| Proves output not tampered | ✅ | ✅ |
| Proves **the model actually ran** | ❌ | ✅ |
| Proves a **specific model version** ran | ❌ | ✅ |
| Hides the input | n/a | ✅ (only a hash is public) |
| Hides the model weights | n/a | ✅ |
| Trust assumption | trust the signer's claim that they ran the model | trust only the math + public params |

**Use signing** when you only need "this output came from us and wasn't
altered" and you trust the issuer to have actually run the model.
**Use zkml-trail** when the threat model includes the issuer lying about
*whether*, or *with which model*, the computation was performed — e.g.
regulators, adversarial counterparties, or audit trails that must hold up
without trusting the agent operator.

---

## For AI engineers

**What it does.** Wraps any ONNX model so each inference emits a verifiable
receipt. Integrate via the CLI or the REST API.

**Integrate (CLI):**

```bash
zkml-trail setup your_model.onnx --name your-agent --calibration repr.json
# in your agent loop, after a decision:
zkml-trail prove --model your_model.onnx --input this_decision.json --output r.json
```

**Integrate (HTTP):**

```bash
zkml-trail serve                       # or: docker compose up
curl -F model=@your_model.onnx -F name=your-agent http://localhost:8000/agents/register
curl -X POST http://localhost:8000/agents/<agent_id>/decide \
     -H 'content-type: application/json' \
     -d '{"decision_input": {"input": [0.82, 0.61, 0.44, 0.50, 0.33]}}'
```

**Performance.** Every command prints proof time, proof size, and
verification time so you can decide whether the ZK overhead is worth it.
For the tiny demo models (logrows 15): setup ≈ 8 s (one-time), prove ≈ 5 s,
proof ≈ 4 KB, verify ≈ 40 ms. Cost scales with model size (circuit
`logrows`); large models can take much longer to prove. Measure your model.

**Model requirements.** Export to ONNX with **static shapes** (no dynamic
axes — a ZK circuit is a fixed arithmetic graph). Linear / Conv / matmul /
common-activation graphs lower well. See the EZKL supported-ops list.

**Calibration matters.** Pass `--calibration <representative_input.json>` at
setup. The circuit's fixed-point scales and lookup tables are sized from
that sample; a degenerate or unrepresentative sample produces a worse (and,
in some EZKL builds, unstable) circuit.

## For security researchers

**Circuit construction.** EZKL lowers the ONNX graph to a Halo2 circuit
over the BN254 scalar field using KZG commitments. zkml-trail fixes the
visibility configuration to:

- `input_visibility = "hashed"` — the private input is committed via an
  in-circuit **Poseidon** hash that is exposed as a *public* input. This is
  what makes `decision.input_hash` cryptographically *bound by the proof*
  rather than merely asserted by the issuer.
- `output_visibility = "public"` — the decision is a public output.
- `param_visibility = "fixed"` — model weights are baked into the circuit
  and verifying key; they never appear in the witness or the receipt.

**The statement a receipt proves**, precisely: *"There exists a private
input `x` with `Poseidon(x) = H` such that running the circuit committed by
verifying key `VK` on `x` yields public output `O`."* Where `VK` is itself a
commitment to the model weights.

**Verifier checks (independent, in `backend.verify`):**

1. the Halo2 proof verifies against `(settings, vk, srs)`;
2. the supplied `vk` hashes to the `vk_hash` bound in the receipt (so you
   can't verify against a substituted key);
3. the supplied `srs` hashes to the `srs_hash` bound in the receipt;
4. the public output inside the proof equals the receipt's stated
   `output`/`decision` (blocks editing the human-readable verdict);
5. the public Poseidon commitment inside the proof equals the receipt's
   `input_hash`.

**Threat model & trust assumptions.**

- **SRS / trusted setup.** Halo2-KZG needs a structured reference string
  from a powers-of-tau ceremony. zkml-trail prefers the canonical
  perpetual-powers-of-tau SRS (`get_srs`). **If that host is unreachable**
  (offline / restricted networks, as in this repo's CI), it falls back to a
  **locally generated SRS** and flags it loudly. A locally generated SRS is
  effectively a *trusted setup by the issuer*: a malicious issuer who knows
  the toxic waste could forge proofs. For adversarial use, ship the
  canonical ceremony SRS; the `srs_hash` binding lets verifiers confirm
  which SRS was used.
- **Quantization.** EZKL proves a *fixed-point* approximation of your
  float model. The receipt attests the quantized computation, which differs
  from the float model by a small, calibrated error (printed at setup). If
  your decision boundary is razor-thin, the quantized and float models can
  disagree.
- **Weights ≠ behavior.** The proof binds the weights via `VK`, but does
  not certify those weights are the "approved" ones — pin and publish the
  expected `model_hash`/`vk_hash` out of band.
- **What's still trusted:** the EZKL/Halo2 implementation, the ONNX→circuit
  lowering, and (in fallback mode) the SRS.

**Attack surface / known limitations.** The prover is a native (Rust)
extension; in constrained containers EZKL's prover can occasionally
deadlock. zkml-trail runs proving in an isolated subprocess with a
wall-clock timeout and bounded retry so a stalled attempt is killed and
retried rather than hanging the service. The verifier is cheap and pure.

## For regulators / lawyers

**What a receipt establishes (evidentiary value):** that a particular
software model — identified by an immutable cryptographic hash — was
executed on some input, and produced a recorded output, at a recorded time,
and that none of these were altered afterward. The proof is verifiable by
*any* third party (regulator, auditor, opposing counsel, insurer) using
only public parameters, with no access to the private input or the
proprietary model. Verification is deterministic mathematics, not a trusted
attestation.

**What it does NOT establish:** that the input data was truthful, that the
model is fair/unbiased/appropriate, or that the decision was lawful or
correct. It is evidence of *faithful, attributable execution*, not of sound
judgment.

**Chain-of-custody properties.** The input is never stored (only its hash
is bound). The output and proof are tamper-evident: any later edit to the
recorded decision invalidates verification. Receipts are independently
re-verifiable indefinitely, so the audit trail does not depend on the
continued existence or honesty of the issuer.

**EU AI Act relevance.** For high-risk AI systems, the Act emphasizes
record-keeping / logging (Art. 12), transparency, and human-oversight
traceability. zkml-trail produces logs that are *cryptographically
non-repudiable* and *privacy-preserving* (no PII in the audit record, only
a hash) — useful for demonstrating that a specific conformity-assessed
model version handled a given case, without exposing the data subject's
information. It complements, and does not replace, the Act's bias, data-
governance, and oversight obligations (which ZK does not address).

## For executives

zkml-trail turns each AI agent decision into a cryptographic receipt that
anyone can verify without trusting you and without seeing your data or your
model. It mitigates the risk that an audit trail is dismissed as
fabricated, that you can't prove an automated decision was actually made by
the approved model, or that demonstrating compliance means exposing
sensitive inputs or proprietary weights. It enables privacy-preserving,
non-repudiable record-keeping for regulated, high-stakes, or adversarial AI
deployments (finance, healthcare, hiring, content moderation, lending).

---

## REST API

Run `zkml-trail serve` (or `docker compose up`), then:

| Method & path | Purpose |
|---|---|
| `POST /agents/register` | Upload an ONNX model → `agent_id`, `vk_key_url`, `model_hash` |
| `POST /agents/{agent_id}/decide` | Run inference privately → `receipt_id`, `receipt`, decision (input not stored) |
| `POST /receipts/verify` | Verify a receipt blob (+ optional vk) |
| `GET  /receipts/{receipt_id}` | Public receipt (input redacted, proof included) |
| `GET  /agents/{agent_id}/vk` | Download the public verifying key |
| `GET  /agents/{agent_id}/srs` | Download the public SRS (needed with the vk) |
| `GET  /agents/{agent_id}/history` | Full audit trail (decisions public, inputs redacted) |

Interactive docs at `http://localhost:8000/docs`.

---

## Examples

| Example | Input (private) | Output (in receipt) | "Prove your agent…" |
|---|---|---|---|
| [`examples/trading-agent`](examples/trading-agent) | market/momentum data | buy / sell / hold | …traded autonomously |
| [`examples/moderation-agent`](examples/moderation-agent) | content features | allow / remove | …moderated this content |
| [`examples/hiring-agent`](examples/hiring-agent) | resume features | score | …scored every candidate with the same model |

Each ships a tiny `model.onnx`, a `sample_input.json`, a pre-generated
`sample_receipt.json`, a one-command `verify.sh`, and a README.

---

## How it works (one diagram)

```
                 ┌─────────────── setup (once per model version) ───────────────┐
  model.onnx ──▶ │ ONNX → Halo2 circuit → proving key (PRIVATE) + verifying key │
                 │ (PUBLIC) + SRS (PUBLIC). model_id = hash(model).             │
                 └──────────────────────────────────────────────────────────────┘

                 ┌─────────────── prove (per decision) ─────────────────────────┐
  input ───────▶ │ run inference privately → witness → ZK proof.                 │
  (never stored) │ public: output + Poseidon(input). private: input, weights.    │ ──▶ receipt.json
                 └──────────────────────────────────────────────────────────────┘

                 ┌─────────────── verify (anyone, forever) ─────────────────────┐
  receipt ─────▶ │ check proof against verifying key + SRS. no model, no input.  │ ──▶ verified: true/false
  + vk + srs     └──────────────────────────────────────────────────────────────┘
```

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [EZKL](https://github.com/zkonduit/ezkl) and the Halo2 proving
system. zkml-trail is the receipt/audit-trail layer on top.
