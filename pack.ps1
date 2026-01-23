# Delete dist if it already exists
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

# Create dist
New-Item "dist" -ItemType Directory

# Install dependencies
if (Test-Path "requirements.txt") {
    pip install --target ./deps -r requirements.txt
}

# Ensure Supertonic 2 models are present (historical layout)
if (-not (Test-Path "model\duration_predictor.onnx")) {
    if (Test-Path "scripts\download_supertonic2_models.ps1") {
        Write-Host "Supertonic 2 model files not found; downloading into .\model ..."
        .\scripts\download_supertonic2_models.ps1
    }
    else {
        throw "Missing model\*.onnx and scripts\download_supertonic2_models.ps1 not found."
    }
}

# Remember to add any additional files, and change the name of the plugin
$artifacts = "cn-plugin-supertonic-tts.py", "requirements.txt", "manifest.json", "__init__.py"

if (Test-Path "deps") {
    $artifacts += "deps"
}

if (Test-Path "model") {
    $artifacts += "model"
}

if (Test-Path "vendor") {
    $artifacts += "vendor"
}

$compress = @{
LiteralPath = $artifacts
CompressionLevel = "Fastest"
DestinationPath = "dist\cn-plugin-supertonic-ttsPlugin.zip" # Change the name of the plugin
}
Compress-Archive @compress