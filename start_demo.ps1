#Requires -Version 7.0
<#
.SYNOPSIS
Start the local single-table demo with portable defaults.

.DESCRIPTION
All paths are resolved from this project directory. Override them with
parameters or the TABLE_VLM_* environment variables; nothing points at the
original author's <DATA_ROOT> directory.

.EXAMPLE
.\start_demo.ps1
.EXAMPLE
.\start_demo.ps1 -Port 7862 -Model D:\models\Qwen3-VL-2B-Instruct -GpuPython .\.venv\Scripts\python.exe
#>
param(
    [int]$Port = 7861,
    [string]$HostName = '127.0.0.1',
    [string]$Python = '',
    [string]$Model = '',
    [string]$Adapter = '',
    [string]$OutputRoot = '',
    [string]$GpuPython = '',
    [string]$ExtraPythonPath = '',
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $Python) {
    if ($env:TABLE_VLM_PYTHON) { $Python = $env:TABLE_VLM_PYTHON }
    elseif (Test-Path -LiteralPath (Join-Path $root '.venv\Scripts\python.exe')) { $Python = Join-Path $root '.venv\Scripts\python.exe' }
    else { $Python = 'python' }
}
if (-not $OutputRoot) {
    $OutputRoot = if ($env:TABLE_VLM_OUTPUT) { $env:TABLE_VLM_OUTPUT } else { Join-Path $root 'demo-output' }
} elseif ($OutputRoot -notmatch '^([A-Za-z]:[\\/]|\\\\|/)') {
    $OutputRoot = Join-Path $root $OutputRoot
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$url = "http://${HostName}:$Port"
$recordFile = Join-Path $OutputRoot 'server.json'

try {
    $existing = Invoke-RestMethod -Uri "$url/config" -TimeoutSec 2
} catch {
    $existing = $null
}
if ($existing) {
    if ($existing.title -ne '表格图片识别') { throw "Port $Port is occupied by another application." }
    if (-not $Force) {
        Write-Output "Demo is already running: $url"
        return
    }
    & (Join-Path $root 'stop_demo.ps1') -OutputRoot $OutputRoot
    Start-Sleep -Seconds 2
}

$arguments = @('-u', '-m', 'table_vlm.demo', '--port', "$Port", '--host', $HostName, '--output-root', $OutputRoot)
if ($Model) { $arguments += @('--model', $Model) }
if ($Adapter) { $arguments += @('--adapter', $Adapter) }
if ($GpuPython) { $arguments += @('--gpu-python', $GpuPython) }
if ($ExtraPythonPath) { $arguments += @('--extra-pythonpath', $ExtraPythonPath) }
$argLine = ($arguments | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
}) -join ' '
$process = Start-Process -FilePath $Python -ArgumentList $argLine -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $OutputRoot 'server-out.log') `
    -RedirectStandardError (Join-Path $OutputRoot 'server-error.log') -PassThru
if ($process.HasExited) {
    throw "Demo exited immediately ($($process.ExitCode)); see $(Join-Path $OutputRoot 'server-error.log')."
}
@{ pid = $process.Id; url = $url; output_root = $OutputRoot
   purpose = 'table-vlm single image demo'; demo_revision = 'a5-min65536-v1'
   process_start_ticks = $process.StartTime.ToUniversalTime().Ticks } |
    ConvertTo-Json | Set-Content -LiteralPath $recordFile -Encoding utf8
Write-Output "Demo is starting: $url"
Write-Output "Logs: $OutputRoot"
Write-Output 'Stop only this demo with .\stop_demo.ps1'
