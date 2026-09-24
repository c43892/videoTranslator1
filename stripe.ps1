param(
    [Parameter(Position=0)][ValidateSet('status','sandbox','test','live')][string]$Mode = 'status',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Mode -eq 'test') { $Mode = 'sandbox' }
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$scriptPath = Join-Path $PSScriptRoot 'deploy/stripe-mode.py'
$listenerPath = Join-Path $PSScriptRoot 'deploy/stripe-listen.py'
$pidFile = Join-Path $PSScriptRoot 'vt-data/stripe-listener.pid'
$logFile = Join-Path $PSScriptRoot 'vt-data/stripe-listener.log'
$errorLog = Join-Path $PSScriptRoot 'vt-data/stripe-listener.err.log'
$envFile = Join-Path $PSScriptRoot '.env'
$backup = Join-Path $PSScriptRoot '.env.stripe-switch.rollback'
$env:PATH = "$env:LOCALAPPDATA/Programs/DockerDesktop/resources/bin;$env:LOCALAPPDATA/Microsoft/WinGet/Links;" + $env:PATH
$compose = @('compose','--env-file',$envFile,'-p','videotranslator-studio','-f','compose.chat.yml','-f','compose.engine.yml')

function Invoke-CheckedPython([string[]]$Arguments) {
    & $python $scriptPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Stripe configuration check failed.' }
}
function Invoke-Compose([string[]]$Arguments) {
    & docker @compose @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Studio container command failed.' }
}
function Stop-ProcessTree([int]$ProcessNumber) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessNumber" | ForEach-Object { Stop-ProcessTree $_.ProcessId }
    Stop-Process -Id $ProcessNumber -ErrorAction SilentlyContinue
}
function Stop-Listener {
    if (Test-Path -LiteralPath $pidFile) {
        $number = [int](Get-Content -LiteralPath $pidFile -Raw)
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$number"
        $legacyLocal = $process -and $process.ExecutablePath -eq $python -and $process.CommandLine -match 'deploy[\\/]stripe-listen\.py'
        if ($process -and ($process.CommandLine -like "*$listenerPath*" -or $legacyLocal)) { Stop-ProcessTree $number }
        Remove-Item -LiteralPath $pidFile
    }
}
function Start-Listener {
    $process = Start-Process -FilePath $python -ArgumentList @('"' + $listenerPath + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $logFile -RedirectStandardError $errorLog -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        if ($process.HasExited) { throw 'Stripe listener exited; inspect vt-data/stripe-listener.err.log' }
        if ((Test-Path -LiteralPath $logFile) -and (Select-String -LiteralPath $logFile -SimpleMatch 'Ready!' -Quiet)) { return }
        Start-Sleep -Seconds 1
    }
    throw 'Stripe listener did not become ready.'
}
function Test-Application([string]$ExpectedMode) {
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        try {
            $config = Invoke-RestMethod 'http://127.0.0.1:8090/api/v1/chat/config' -TimeoutSec 5
            if ($config.payment_mode -eq $ExpectedMode -and $config.payment_providers -contains 'stripe') { return }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw 'Studio did not report the expected Stripe environment.'
}

if ($Mode -eq 'status') {
    Invoke-CheckedPython @('status')
    try {
        $running = Invoke-RestMethod 'http://127.0.0.1:8090/api/v1/chat/config' -TimeoutSec 5
        Write-Output "Running Stripe mode: $($running.payment_mode)"
    } catch { Write-Output 'Studio is not reachable.' }
    exit 0
}
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'vt-data') -Force | Out-Null
$lock = [System.IO.File]::Open((Join-Path $PSScriptRoot 'vt-data/stripe-switch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$stopped = $false
try {
    Invoke-CheckedPython @('check',$Mode)
    if ($CheckOnly) { exit 0 }
    # Build first, while the existing application is still serving requests.
    Invoke-Compose @('build','studio')
    Copy-Item -LiteralPath $envFile -Destination $backup -Force
    Invoke-Compose @('stop','studio')
    $stopped = $true
    # Recheck after stopping request admission to close the preflight race.
    Invoke-CheckedPython @('check',$Mode)
    Stop-Listener
    Invoke-CheckedPython @('set',$Mode)
    Start-Listener
    Invoke-Compose @('up','-d','--no-deps','--no-build','studio')
    Test-Application $Mode
    Write-Output "Stripe is now $Mode. Refresh the browser. Balances and task histories remain separate."
} catch {
    $failure = $_
    if ($stopped) {
        Write-Warning 'Switch failed. Restoring the previous configuration and application.'
        Stop-Listener
        Copy-Item -LiteralPath $backup -Destination $envFile -Force
        try { Start-Listener } catch { Write-Warning 'Previous listener needs attention; inspect its log.' }
        Invoke-Compose @('up','-d','--no-deps','--no-build','studio')
    }
    throw $failure
} finally {
    $lock.Dispose()
}
