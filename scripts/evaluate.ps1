# Evaluate trained runs on the held-out test split and regenerate the comparison.
#   scripts\evaluate.ps1                       # both runs + comparison
#   scripts\evaluate.ps1 -Run rdd_yolo26n      # single run
param([string]$Run = "", [ValidateSet("val", "test")][string]$Split = "test")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

$runs = if ($Run) { @($Run) } else { @("baseline_yolo26n", "rdd_yolo26n") }
foreach ($r in $runs) {
    & $Py training\evaluate.py --run $r --split $Split
    if ($LASTEXITCODE -ne 0) { throw "Evaluation of $r failed" }
}
if (-not $Run) { & $Py training\compare.py --baseline baseline_yolo26n --rdd rdd_yolo26n --split $Split }
