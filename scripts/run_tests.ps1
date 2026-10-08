# Run the backend/ML test suite and the frontend type-check + production build.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "==> pytest" -ForegroundColor Cyan
& $Py -m pytest tests -q
$py_rc = $LASTEXITCODE

Write-Host "==> frontend build (tsc + vite)" -ForegroundColor Cyan
Push-Location frontend
npm run build
$fe_rc = $LASTEXITCODE
Pop-Location

if ($py_rc -ne 0 -or $fe_rc -ne 0) { Write-Host "FAILED (pytest=$py_rc, frontend=$fe_rc)" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
