#Requires -Version 7.0
<#
.SYNOPSIS
Create a project-local .venv and prepare the demo on Windows.

.DESCRIPTION
Installs the CUDA 12.4 build of torch 2.6.0, the demo/data/worker
dependencies, downloads the pinned 4.25GB base weight (unless already present),
and runs the preflight. Nothing outside this project directory is modified
except the pip/Hugging Face caches.

.EXAMPLE
.\scripts\setup_demo_windows.ps1
.EXAMPLE
.\scripts\setup_demo_windows.ps1 -Python C:\Python312\python.exe -SkipTorch
#>
param(
    [string]$Python = '',
    [switch]$SkipTorch,
    [switch]$SkipDownload,
    [string]$Transport = 'modelscope'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not $Python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) { throw 'python was not found; pass -Python C:\path\to\python.exe' }
    $Python = $command.Source
}
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host '==> creating .venv'
    & $Python -m venv $venv
}
& $venvPython -m pip install --upgrade pip
if (-not $SkipTorch) {
    Write-Host '==> installing torch 2.6.0 + cu124 (about 2.5GB)'
    & $venvPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
}
Write-Host '==> installing demo, data and worker dependencies'
& $venvPython -m pip install -r requirements-data.txt -r requirements-demo.txt `
    transformers==4.57.1 peft==0.18.1 accelerate==1.11.0
if (-not $SkipDownload) {
    $weight = Join-Path $root 'models\Qwen3-VL-2B-Instruct\model.safetensors'
    if (Test-Path -LiteralPath $weight) {
        Write-Host '==> base weight already present; verifying'
    } else {
        Write-Host '==> downloading the pinned base weight (about 4.25GB)'
        & $venvPython -m table_vlm.download_model `
            --output 'models\Qwen3-VL-2B-Instruct' `
            --metadata-manifest 'research\qwen2b-fixed-manifest.json' `
            --transport $Transport
    }
}
Write-Host '==> preflight'
& $venvPython -m table_vlm.doctor --deep --require-cuda
Write-Host ''
Write-Host 'Ready. Start the UI with: .\start_demo.ps1'
