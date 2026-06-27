# moderation-agent — *prove your agent moderated*

A tiny content classifier. It takes content features and outputs a
probability of **remove**, decoded as a binary **allow / remove** decision.

- **Input (private):** 8 content features (toxicity, slur count, threat
  score, harassment, positivity, factuality, caps ratio, link spam). See
  `sample_input.json`.
- **Output (in receipt):** `allow` or `remove` (threshold 0.5).
- **The claim a receipt proves:** *this content was flagged by this exact
  model running on the real content features — not by a human with a
  political bias, and not by a different model.* The content stays private;
  only the decision and the input's hash are public. Useful as evidence in
  an **appeals process**: the platform can show the decision was automated
  and reproducible without republishing the (possibly sensitive) content.

## Run it

```bash
pip install -e ../..                 # from repo root, once
python ../../scripts/make_example_models.py   # if model.onnx is missing
./verify.sh
```

Then inspect what's proven vs. private:

```bash
zkml-trail inspect receipt.json
```

## Files

| File | What it is |
|---|---|
| `model.onnx` | tiny demo classifier (length-8 input → sigmoid) |
| `sample_input.json` | example private input (toxic content → remove) |
| `sample_receipt.json` | a pre-generated receipt, for inspection |
| `sample_vk.key` | the public verifying key for that receipt |
| `verify.sh` | one command: setup → prove → verify |

> The honest caveat: a receipt proves the *model executed faithfully*. It
> does **not** prove the moderation policy is correct or unbiased — ZK
> proves execution, not fairness. See the repo README › limitations.
