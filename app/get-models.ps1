# Downloads all ONNX models into assets/models/ for the Recall Flutter app. Idempotent.
# Run: powershell -File get-models.ps1
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dl = Join-Path $root '.models-dl'
$m = Join-Path $root 'assets\models'
New-Item -ItemType Directory -Force $dl, "$m\whisper", "$m\vad", "$m\speaker", "$m\embed" | Out-Null

if (-not (Test-Path "$m\vad\silero_vad.onnx")) {
    "downloading silero vad..."
    curl.exe -sSfL 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx' -o "$m\vad\silero_vad.onnx"
}

if (-not (Test-Path "$m\speaker\model.onnx")) {
    "downloading speaker model..."
    curl.exe -sSfL 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx' -o "$m\speaker\model.onnx"
}

if (-not (Test-Path "$m\whisper\encoder.onnx")) {
    "downloading whisper tiny.en (int8)..."
    curl.exe -sSfL 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2' -o "$dl\whisper.tar.bz2"
    tar -xf "$dl\whisper.tar.bz2" -C $dl
    $d = "$dl\sherpa-onnx-whisper-tiny.en"
    Copy-Item "$d\tiny.en-encoder.int8.onnx" "$m\whisper\encoder.onnx"
    Copy-Item "$d\tiny.en-decoder.int8.onnx" "$m\whisper\decoder.onnx"
    Copy-Item "$d\tiny.en-tokens.txt" "$m\whisper\tokens.txt"
}

if (-not (Test-Path "$m\embed\model.onnx")) {
    "downloading bge-small-en-v1.5..."
    curl.exe -sSfL 'https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx' -o "$m\embed\model.onnx"
    curl.exe -sSfL 'https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/vocab.txt' -o "$m\embed\vocab.txt"
}
"all models ready"
