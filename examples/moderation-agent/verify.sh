#!/usr/bin/env bash
# One command to issue and verify a receipt for the moderation-agent example.
# This runs the *real* ZK pipeline end-to-end (setup -> prove -> verify),
# so a green run is a genuinely verified zero-knowledge proof.
set -euo pipefail
cd "$(dirname "$0")"
export ZKML_TRAIL_HOME="$(pwd)/.zkml-trail"

echo "[1/3] setup  — compiling ZK circuit + keys (one-time) ..."
zkml-trail setup model.onnx --name "moderation-agent" --task "binary" --labels "allow,remove" --calibration sample_input.json

echo "[2/3] prove  — running inference privately + issuing receipt ..."
zkml-trail prove --model model.onnx --input sample_input.json --output receipt.json

VK="$(ls "$ZKML_TRAIL_HOME"/agents/*/vk.key | head -1)"
echo "[3/3] verify — checking the proof with only the public verifying key ..."
zkml-trail verify receipt.json --vk "$VK"

echo
echo "Inspect it:  zkml-trail inspect receipt.json"
