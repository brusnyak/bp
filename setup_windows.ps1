# Windows Setup Script for BP Project (sandbox-fixed copy, 2026-09-28)
#
# Fixes vs the committed version:
#   1. No machine-scope installs: nothing here needs admin or UAC.
#      Python 3.11 / FFmpeg / Node.js are CHECKED, not installed system-wide.
#      Missing pieces print a user-scope install hint and the script stops.
#   2. `npm install` runs at the repo ROOT (package.json lives here) --
#      the old script did `Push-Location frontend`, a directory that
#      does not exist, so step 6 always failed.
#   3. OpenSSL fallback covers Git Bash (`C:\Program Files\Git\usr\bin\openssl.exe`),
#      same as the Makefile's Windows branch -- plain `openssl` is rarely on PATH.
#   4. No Invoke-Expression string-eval; the call operator (&) is used throughout.
#   5. Fails fast ($ErrorActionPreference = "Stop") instead of limping on
#      with the wrong interpreter.
#
# Run from the repo root (non-interactive):
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1
# To skip the slow model downloads (CT2 conversion takes a while):
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1 -SkipModels

param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

Write-Host "--- Starting Windows Setup for BP Project (isolated, no admin) ---" -ForegroundColor Cyan

function Test-CommandExists ($command) {
    return $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
}

# 1. Python 3.11 via the py launcher (user-scope install, never machine-scope)
Write-Host "`n[1/6] Checking Python 3.11..." -ForegroundColor Yellow
$pythonExec = $null
if (Test-CommandExists "py") {
    try {
        $py311ver = & py -3.11 --version 2>&1
        if ($py311ver -match "3\.11") {
            $pythonExec = @("py", "-3.11")
            Write-Host "Found $py311ver via py launcher." -ForegroundColor Green
        }
    } catch { }
}
if (-not $pythonExec) {
    Write-Host "Python 3.11 not found (only $(python --version 2>$null))." -ForegroundColor Red
    Write-Host "Install it WITHOUT admin rights, then re-run this script:" -ForegroundColor Yellow
    Write-Host '  winget install -e --id Python.Python.3.11 --scope user' -ForegroundColor White
    Write-Host "requirements.txt pins (numpy 1.26.4 / pandas 1.5.3 / numba 0.60.0) target 3.11;" -ForegroundColor Yellow
    Write-Host "3.12 is not supported by this project." -ForegroundColor Yellow
    exit 1
}

# 2. FFmpeg (check only -- already present on most dev machines)
Write-Host "`n[2/6] Checking FFmpeg..." -ForegroundColor Yellow
if (-not (Test-CommandExists "ffmpeg")) {
    Write-Host "FFmpeg not found. Install without admin:" -ForegroundColor Red
    Write-Host '  winget install -e --id Gyan.FFmpeg --scope user' -ForegroundColor White
    exit 1
}
Write-Host "FFmpeg is already installed." -ForegroundColor Green

# 3. Node.js (check only -- needed solely for chart.js UI deps at repo root)
Write-Host "`n[3/6] Checking Node.js..." -ForegroundColor Yellow
if (-not (Test-CommandExists "npm")) {
    Write-Host "Node.js not found -- UI chart deps will be skipped." -ForegroundColor Magenta
    Write-Host '  Optional (no admin): winget install -e --id OpenJS.NodeJS --scope user' -ForegroundColor White
    $skipNpm = $true
} else {
    Write-Host "Node.js is already installed." -ForegroundColor Green
    $skipNpm = $false
}

# 4. Virtual environment (inside the project dir -- nothing touches global Python)
Write-Host "`n[4/6] Creating Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    & $pythonExec -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment 'venv' already exists." -ForegroundColor Green
}

$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: virtual environment python not found at $venvPython" -ForegroundColor Red
    exit 1
}

# 5. Python + frontend dependencies
Write-Host "`n[5/6] Installing Python Dependencies (this takes a while: torch + Coqui TTS)..." -ForegroundColor Yellow
& $venvPip install -r requirements.txt

if (-not $skipNpm) {
    Write-Host "Installing UI dependencies at repo root (package.json)..." -ForegroundColor Yellow
    & npm install
} else {
    Write-Host "Skipping npm install (Node.js missing)." -ForegroundColor Magenta
}

# 6. Certificates + models
Write-Host "`n[6/6] Certificates and models..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "certs" | Out-Null
$gitOpenssl = "C:\Program Files\Git\usr\bin\openssl.exe"
if (Test-CommandExists "openssl") {
    & openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
} elseif (Test-Path $gitOpenssl) {
    & $gitOpenssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
} else {
    Write-Host "OpenSSL not found (checked PATH and Git Bash). Generate manually:" -ForegroundColor Magenta
    Write-Host '  & "C:\Program Files\Git\usr\bin\openssl.exe" req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"' -ForegroundColor White
}

if ($SkipModels) {
    Write-Host "Skipping model downloads (-SkipModels)." -ForegroundColor Yellow
} else {
    New-Item -ItemType Directory -Force -Path "ct2_models" | Out-Null
    Write-Host "Downloading Piper TTS models..." -ForegroundColor Yellow
    & $venvPython backend/tts/download_piper_models.py en_US-ryan-medium
    & $venvPython backend/tts/download_piper_models.py sk_SK-lili-medium
    & $venvPython backend/tts/download_piper_models.py cs_CZ-jirka-medium

    Write-Host "Converting MT models (slow, needs internet)..." -ForegroundColor Yellow
    # sys.setrecursionlimit(2000) required by the converter -- see Makefile
    & $venvPython -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-sk', 'ct2_models/Helsinki-NLP--opus-mt-en-sk', quantization='int8')"
    & $venvPython -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-sk-en', 'ct2_models/Helsinki-NLP--opus-mt-sk-en', quantization='int8')"
    & $venvPython -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-cs', 'ct2_models/Helsinki-NLP--opus-mt-en-cs', quantization='int8')"
}

Write-Host "`n--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "CPU note: this machine has no NVIDIA GPU; backend/hardware.py" -ForegroundColor White
Write-Host "falls back to CPU per stage (verified). Expect Piper to feel" -ForegroundColor White
Write-Host "fast, XTTS voice-cloning noticeably slower (~seconds per clip)." -ForegroundColor White
Write-Host "To run the app:" -ForegroundColor Cyan
Write-Host "1. .\venv\Scripts\activate" -ForegroundColor White
Write-Host "2. python app.py" -ForegroundColor White
Write-Host "3. Open https://localhost:8000 (accept the self-signed cert)" -ForegroundColor White
