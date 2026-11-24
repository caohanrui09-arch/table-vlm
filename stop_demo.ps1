#Requires -Version 7.0
<#
.SYNOPSIS
Stop only the demo process recorded by start_demo.ps1.
#>
param(
    [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRoot) {
    if ($env:TABLE_VLM_OUTPUT) { $OutputRoot = $env:TABLE_VLM_OUTPUT }
    else { $OutputRoot = Join-Path $PSScriptRoot 'demo-output' }
} elseif ($OutputRoot -notmatch '^([A-Za-z]:[\\/]|\\\\|/)') {
    $OutputRoot = Join-Path $PSScriptRoot $OutputRoot
}
$recordFile = Join-Path $OutputRoot 'server.json'
if (-not (Test-Path -LiteralPath $recordFile)) { throw 'Demo process record not found.' }
$record = Get-Content -LiteralPath $recordFile -Raw | ConvertFrom-Json
if ($record.purpose -ne 'table-vlm single image demo' -or -not $record.process_start_ticks) {
    throw 'Process ownership record is incomplete; nothing was stopped.'
}
$demoProcess = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
if (-not $demoProcess) { Write-Output 'Demo is already stopped.'; return }
if ($demoProcess.StartTime.ToUniversalTime().Ticks -ne [long]$record.process_start_ticks) {
    throw 'Process ID was reused; nothing was stopped.'
}
# Stop only the recorded demo launcher and its child inference process, if any.
& taskkill.exe /PID $demoProcess.Id /T /F
if ($LASTEXITCODE -ne 0) { throw "Could not stop the recorded demo: $LASTEXITCODE" }
