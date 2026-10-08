# Train one experiment, or the full pipeline (baseline + RDD-YOLO + evaluation + comparison).
#   scripts\train.ps1                          # full pipeline, 40 epochs each (resumable)
#   scripts\train.ps1 -Experiment rdd -Epochs 60
#   scripts\train.ps1 -Smoke                   # 1-epoch sanity check on 2% of the data
param([ValidateSet("all", "baseline", "rdd")][string]$Experiment = "all", [int]$Epochs = 40, [int]$Batch = 16,
      [int]$Workers = 6, [switch]$Smoke)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if ($Smoke) {
    & $Py training\train.py --experiment rdd --epochs 1 --fraction 0.02 --name smoke_rdd --batch $Batch --workers 2
} elseif ($Experiment -eq "all") {
    & $Py scripts\run_experiments.py --epochs $Epochs --batch $Batch --workers $Workers
} else {
    & $Py scripts\run_experiments.py --only $Experiment --epochs $Epochs --batch $Batch --workers $Workers
}
exit $LASTEXITCODE
