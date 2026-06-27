# trading-agent — *prove your agent traded*

A tiny momentum trading model. It takes recent normalized price-momentum
features and outputs **sell / hold / buy**.

- **Input (private):** 5 momentum features — `model.onnx` expects a length-5
  vector. See `sample_input.json`.
- **Output (in receipt):** a 3-way decision, decoded by argmax.
- **The claim a receipt proves:** *the trade decision was produced by this
  exact model (by hash) running on a real, private market-data input — not
  by a human manually trading, and not by a different model.* The market
  data stays private; only the decision and the input's hash are public.

## Run it

```bash
pip install -e ../..                 # from repo root, once
python ../../scripts/make_example_models.py   # if model.onnx is missing
./verify.sh
```

`verify.sh` runs the real pipeline — compile circuit → prove → verify —
and prints `verified: true`. Then:

```bash
zkml-trail inspect receipt.json
```

## Files

| File | What it is |
|---|---|
| `model.onnx` | tiny demo momentum model (length-5 input → 3 logits) |
| `sample_input.json` | example private input (bullish momentum → buy) |
| `sample_receipt.json` | a pre-generated receipt, for inspection |
| `sample_vk.key` | the public verifying key for that receipt |
| `verify.sh` | one command: setup → prove → verify |

> Note: `sample_receipt.json` was generated with a locally generated SRS
> (the canonical ceremony host is unreachable in CI). To re-verify from
> scratch with a consistent SRS, run `./verify.sh`, which regenerates the
> bundle and verifies a fresh receipt. See the repo README › threat model.
