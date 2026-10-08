# Download (selectively) and prepare the RDD2022 dataset.
#   scripts\prepare_data.ps1                         # default 5 countries (~2.4 GB download)
#   scripts\prepare_data.ps1 -Countries Czech,India  # smaller subset
#   scripts\prepare_data.ps1 -ValidateOnly           # only re-run validation on existing raw data
param([string[]]$Countries = @("Japan", "India", "Czech", "United_States", "China_MotorBike"), [switch]$ValidateOnly, [switch]$SkipDownload)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not $SkipDownload -and -not $ValidateOnly) {
    Write-Host "==> Downloading RDD2022 ($($Countries -join ', ')) from figshare" -ForegroundColor Cyan
    & $Py dataset\download_rdd2022.py --countries @Countries
    if ($LASTEXITCODE -ne 0) { throw "Download failed (re-run to resume)" }
}

Write-Host "==> Validating and converting to YOLO format" -ForegroundColor Cyan
$args_ = @("dataset\prepare_dataset.py", "--countries") + $Countries
if ($ValidateOnly) { $args_ += "--validate-only" }
& $Py @args_
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed" }
Write-Host "Dataset ready: dataset\processed\rdd2022_yolo\data.yaml" -ForegroundColor Green
