# One-command demo startup.
#   scripts\start_all.ps1          # build the dashboard once and serve everything from the backend (port 8000)
#   scripts\start_all.ps1 -Dev     # backend + Vite dev server (hot reload) in two windows
param([switch]$Dev, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\setup.ps1 first" }

if (-not (Get-ChildItem "models\weights\*_best.pt" -ErrorAction SilentlyContinue)) {
    Write-Warning "No trained checkpoint in models\weights - the dashboard will start but inference is disabled until training finishes."
}

if ($Dev) {
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Root\scripts\start_backend.ps1`""
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Root\scripts\start_frontend.ps1`""
    $url = "http://localhost:5173"
} else {
    Push-Location frontend
    if (-not (Test-Path "node_modules")) { npm install }
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Frontend build failed" }
    Pop-Location
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Root\scripts\start_backend.ps1`""
    $url = "http://localhost:8000"
}

Write-Host "Waiting for the API ..." -ForegroundColor Cyan
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 }
}
Write-Host "RDD-YOLO is running at $url  (API docs: http://127.0.0.1:8000/docs)" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process $url }
