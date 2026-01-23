Param(
  [string]$RootDir = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

$ModelDir = Join-Path $RootDir "model"
$VoiceDir = Join-Path $ModelDir "voices"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $VoiceDir | Out-Null

$baseOnnx = "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx"
$baseVoice = "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles"

$onnxFiles = @(
  "duration_predictor.onnx",
  "text_encoder.onnx",
  "vector_estimator.onnx",
  "vocoder.onnx",
  "tts.json",
  "unicode_indexer.json"
)

$voiceFiles = @(
  "M1.json",
  "M2.json",
  "F1.json",
  "F2.json"
)

Write-Host "Downloading Supertonic 2 ONNX files to $ModelDir"
foreach ($file in $onnxFiles) {
  Write-Host "- $file"
  Invoke-WebRequest -Uri "$baseOnnx/$file" -OutFile (Join-Path $ModelDir $file)
}

Write-Host "Downloading Supertonic 2 voice styles to $VoiceDir"
foreach ($file in $voiceFiles) {
  Write-Host "- $file"
  Invoke-WebRequest -Uri "$baseVoice/$file" -OutFile (Join-Path $VoiceDir $file)
}

Write-Host "Done."
