# hiring-agent — *prove your agent scored fairly*

A tiny resume scorer. It takes resume features and outputs a **score**
(roughly 0–100).

- **Input (private):** 6 resume features (years experience, education,
  skills match, leadership, employment gaps, referral). See
  `sample_input.json`.
- **Output (in receipt):** a numeric score.
- **The claim a receipt proves:** *every candidate was scored by the **same
  model** (identical weights, identical hash) — not a different or tweaked
  model per candidate — running on that candidate's real features.* The
  resume stays private; only the score and the input's hash are public.
  Issue one receipt per candidate and the set is cryptographic evidence
  that scoring was consistent across applicants — useful as **legal
  protection against disparate-treatment claims**.

## Run it

```bash
pip install -e ../..                 # from repo root, once
python ../../scripts/make_example_models.py   # if model.onnx is missing
./verify.sh
```

Then:

```bash
zkml-trail inspect receipt.json
```

## Files

| File | What it is |
|---|---|
| `model.onnx` | tiny demo scorer (length-6 input → score) |
| `sample_input.json` | example private input (strong candidate) |
| `sample_receipt.json` | a pre-generated receipt, for inspection |
| `sample_vk.key` | the public verifying key for that receipt |
| `verify.sh` | one command: setup → prove → verify |

> Important honesty: proving *the same model ran for everyone* is **not**
> the same as proving *the model is fair*. zkml-trail gives you consistency
> and non-repudiation; bias testing is a separate obligation. See the repo
> README › limitations.
