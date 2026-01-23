#!/usr/bin/env bash
set -euo pipefail

# Downloads Supertonic 2 ONNX models + voice styles into ./model using the
# *existing* plugin layout:
#   - model/*.onnx, model/tts.json, model/unicode_indexer.json
#   - model/voices/*.json
# Requires: wget

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/model"
VOICE_DIR="$MODEL_DIR/voices"

mkdir -p "$MODEL_DIR" "$VOICE_DIR"

base_onnx="https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx"
base_voice="https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles"

onnx_files=(
  duration_predictor.onnx
  text_encoder.onnx
  vector_estimator.onnx
  vocoder.onnx
  tts.json
  unicode_indexer.json
)

voice_files=(
  M1.json
  M2.json
  F1.json
  F2.json
)

echo "Downloading Supertonic 2 ONNX files to $MODEL_DIR"
for f in "${onnx_files[@]}"; do
  echo "- $f"
  wget -q --show-progress -O "$MODEL_DIR/$f" "$base_onnx/$f"
done

echo "Downloading Supertonic 2 voice styles to $VOICE_DIR"
for f in "${voice_files[@]}"; do
  echo "- $f"
  wget -q --show-progress -O "$VOICE_DIR/$f" "$base_voice/$f"
done

echo "Done."
