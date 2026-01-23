#!/bin/bash

# Delete dist if it already exists
if [ -d "dist" ]; then
    rm -rf dist
fi

# Create dist
mkdir dist

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install --target ./deps -r requirements.txt
fi

# Ensure Supertonic 2 models are present (historical layout)
if [ ! -f "model/duration_predictor.onnx" ]; then
    if [ -f "scripts/download_supertonic2_models.sh" ]; then
        echo "Supertonic 2 model files not found; downloading into ./model ..."
        chmod +x scripts/download_supertonic2_models.sh
        ./scripts/download_supertonic2_models.sh
    else
        echo "Missing model/*.onnx and scripts/download_supertonic2_models.sh not found." >&2
        exit 1
    fi
fi

# Remember to add any additional files, and change the name of the plugin
artifacts=(
    "cn-plugin-supertonic-tts.py"
    "requirements.txt"
    "manifest.json" "__init__.py"
)

if [ -d "deps" ]; then
    artifacts+=("deps")
fi

if [ -d "model" ]; then
    artifacts+=("model")
fi

if [ -d "vendor" ]; then
    artifacts+=("vendor")
fi

# Create the zip archive
zip -r -9 "dist/cn-plugin-supertonic-tts.zip" "${artifacts[@]}"
