# Command-line inference on an image, a folder of images, or a video.
#   scripts\infer.ps1 -Source path\to\road.jpg
#   scripts\infer.ps1 -Source path\to\folder -Conf 0.3
#   scripts\infer.ps1 -Source path\to\dashcam.mp4 -Stride 3
param([Parameter(Mandatory = $true)][string]$Source, [string]$Weights = "", [double]$Conf = 0.25, [int]$Stride = 1)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$a = @((Join-Path $Root "inference\predict.py"), "--source", $Source, "--conf", $Conf, "--stride", $Stride)
if ($Weights) { $a += @("--weights", $Weights) }
& $Py @a
exit $LASTEXITCODE
