# Experiment C: download the extra datasets, build the extended set and fine-tune RDD-YOLO on the GPU.
#   scripts\train_extended.ps1               # ~10.6 GB Norway + ~0.4 GB external downloads, then 25 epochs
param([int]$Epochs = 25, [int]$Batch = 16)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py dataset\download_rdd2022.py --countries China_Drone Norway; if ($LASTEXITCODE -ne 0) { throw "RDD2022 download failed (re-run to resume)" }
& $Py dataset\download_external.py; if ($LASTEXITCODE -ne 0) { throw "External dataset download failed" }
& $Py scripts\run_extended.py --epochs $Epochs --batch $Batch
if ($LASTEXITCODE -eq 0) { & $Py scripts\write_experiments_doc.py }
