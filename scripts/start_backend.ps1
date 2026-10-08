# Start the FastAPI backend on http://127.0.0.1:8000 (API docs at /docs).
param([int]$Port = 8000, [switch]$Reload)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$a = @("-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", $Port)
if ($Reload) { $a += @("--reload", "--reload-dir", "backend", "--reload-dir", "rdd_yolo") }
& $Py @a
