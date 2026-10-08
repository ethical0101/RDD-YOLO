# RDD-YOLO environment setup (Windows PowerShell 5.1+ / PowerShell 7).
# Creates .venv, installs CUDA PyTorch (or CPU build), Python deps and the frontend.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # auto-detect GPU
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu     # force CPU build
param([switch]$Cpu, [string]$Python = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Locating Python 3.11+"
if (-not $Python) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $Python = "py -3.13" } else { $Python = "python" }
}
Invoke-Expression "$Python --version"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Step "Creating virtual environment (.venv)"
    Invoke-Expression "$Python -m venv .venv"
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip | Out-Null

$HasGpu = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    $HasGpu = $LASTEXITCODE -eq 0
}

Step "Installing PyTorch ($(if ($HasGpu) {'CUDA 13.0'} else {'CPU'}))"
$Index = if ($HasGpu) { "https://download.pytorch.org/whl/cu130" } else { "https://download.pytorch.org/whl/cpu" }
& $Py -m pip install torch==2.14.1 torchvision==0.29.1 --index-url $Index
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed (if the download keeps failing, retry - pip resumes)" }

Step "Installing Python requirements"
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

Step "Verifying PyTorch / CUDA"
& $Py -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| ultralytics', ultralytics.__version__)"
& $Py -m rdd_yolo.hardware

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env from .env.example" }

Step "Installing frontend dependencies"
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location frontend; npm install; Pop-Location
} else {
    Write-Warning "npm not found - install Node.js 20.19+ to build the dashboard"
}

Write-Host "`nSetup complete. Next: scripts\prepare_data.ps1" -ForegroundColor Green
