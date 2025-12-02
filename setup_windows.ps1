# Windows Setup Script for BP Project

Write-Host "--- Starting Windows Setup for BP Project ---" -ForegroundColor Cyan

# Function to check if a command exists
function Test-Command ($command) {
    return $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
}

# 1. Check/Install Python 3.11
Write-Host "`n[1/7] Checking Python 3.11..." -ForegroundColor Yellow
if (-not (Test-Command "python")) {
    Write-Host "Python not found. Installing Python 3.11..."
    winget install -e --id Python.Python.3.11 --scope machine
    # Refresh env
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}
else {
    $pyVer = python --version
    if ($pyVer -match "3.11") {
        Write-Host "Python 3.11 is already installed." -ForegroundColor Green
    }
    else {
        Write-Host "Current Python version is $pyVer. Installing Python 3.11 side-by-side..."
        winget install -e --id Python.Python.3.11 --scope machine
        # We will need to find the specific python 3.11 executable
    }
}

# Find Python 3.11 executable
$py311 = Get-Command "py" -ErrorAction SilentlyContinue
if ($py311) {
    $pythonExec = "py -3.11"
}
else {
    # Fallback to just python if it is 3.11, otherwise warn
    if ((python --version) -match "3.11") {
        $pythonExec = "python"
    }
    else {
        Write-Host "Could not find 'py' launcher. Please ensure Python 3.11 is installed and in PATH." -ForegroundColor Red
        # Try to guess path
        $potentialPath = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
        if (Test-Path $potentialPath) {
            $pythonExec = $potentialPath
        }
        else {
            $potentialPathSystem = "C:\Program Files\Python311\python.exe"
            if (Test-Path $potentialPathSystem) {
                $pythonExec = $potentialPathSystem
            }
            else {
                Write-Host "Using default 'python' ($pyVer) - this might fail if not compatible." -ForegroundColor Magenta
                $pythonExec = "python"
            }
        }
    }
}

Write-Host "Using Python executable: $pythonExec" -ForegroundColor Cyan

# 2. Check/Install FFmpeg
Write-Host "`n[2/7] Checking FFmpeg..." -ForegroundColor Yellow
if (-not (Test-Command "ffmpeg")) {
    Write-Host "FFmpeg not found. Installing via winget..."
    winget install -e --id Gyan.FFmpeg
    # Refresh env
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}
else {
    Write-Host "FFmpeg is already installed." -ForegroundColor Green
}

# 3. Check Node.js
Write-Host "`n[3/7] Checking Node.js..." -ForegroundColor Yellow
if (-not (Test-Command "npm")) {
    Write-Host "Node.js not found. Installing..."
    winget install -e --id OpenJS.NodeJS
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}
else {
    Write-Host "Node.js is already installed." -ForegroundColor Green
}

# 4. Create Virtual Environment
Write-Host "`n[4/7] Creating Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Invoke-Expression "$pythonExec -m venv venv"
    Write-Host "Virtual environment created." -ForegroundColor Green
}
else {
    Write-Host "Virtual environment 'venv' already exists." -ForegroundColor Green
}

# 5. Install Python Dependencies
Write-Host "`n[5/7] Installing Python Dependencies..." -ForegroundColor Yellow
# Activate venv for the script execution
$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment python not found at $venvPython" -ForegroundColor Red
    exit 1
}

Invoke-Expression "& '$venvPip' install -r requirements.txt"

# 6. Install Frontend Dependencies
Write-Host "`n[6/7] Installing Frontend Dependencies..." -ForegroundColor Yellow
Push-Location frontend
npm install
Pop-Location

# 7. Download Models & Setup
Write-Host "`n[7/7] Downloading Models and Setting up..." -ForegroundColor Yellow

# Create certs directory
New-Item -ItemType Directory -Force -Path "certs" | Out-Null
# Generate dummy certs if openssl exists, else warn
if (Test-Command "openssl") {
    openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
}
else {
    Write-Host "OpenSSL not found. Skipping certificate generation. You may need to generate 'certs/cert.pem' and 'certs/key.pem' manually or install Git Bash which includes OpenSSL." -ForegroundColor Magenta
}

# Download Piper Models
Write-Host "Downloading Piper TTS models..."
Invoke-Expression "& '$venvPython' backend/tts/download_piper_models.py en_US-ryan-medium"
Invoke-Expression "& '$venvPython' backend/tts/download_piper_models.py sk_SK-lili-medium"
Invoke-Expression "& '$venvPython' backend/tts/download_piper_models.py cs_CZ-jirka-medium"

# Convert MT Models (This might take a while and requires internet)
Write-Host "Converting MT models (this may take time)..."
# We need to set recursion limit as per Makefile
$convertScript = "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-sk', 'ct2_models/Helsinki-NLP--opus-mt-en-sk', quantization='int8')"
Invoke-Expression "& '$venvPython' -c `"$convertScript`""

$convertScript2 = "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-sk-en', 'ct2_models/Helsinki-NLP--opus-mt-sk-en', quantization='int8')"
Invoke-Expression "& '$venvPython' -c `"$convertScript2`""

$convertScript3 = "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-cs', 'ct2_models/Helsinki-NLP--opus-mt-en-cs', quantization='int8')"
Invoke-Expression "& '$venvPython' -c `"$convertScript3`""


Write-Host "`n--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "To run the app:"
Write-Host "1. .\venv\Scripts\activate"
Write-Host "2. python app.py"
Write-Host "`nNote: VB-CABLE was not installed automatically. Please install it manually if needed."
