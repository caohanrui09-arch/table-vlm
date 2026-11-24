#Requires -Version 7.0
<#
.SYNOPSIS
Run a table_vlm module with the project on PYTHONPATH.

.EXAMPLE
.\run_inference.ps1 -m table_vlm.doctor
.EXAMPLE
.\run_inference.ps1 -Python .\.venv\Scripts\python.exe -m table_vlm.demo --image .\examples\table.png
#>
param(
    [string]$Python = '',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not $Python) {
    if ($env:TABLE_VLM_GPU_PYTHON) { $Python = $env:TABLE_VLM_GPU_PYTHON }
    elseif (Test-Path -LiteralPath (Join-Path $projectRoot '.venv\Scripts\python.exe')) {
        $Python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    } else { $Python = 'python' }
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $projectRoot
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command exited with $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
