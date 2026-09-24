$ErrorActionPreference = 'Stop'
$workerRoot = $PSScriptRoot
$userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute (Join-Path $workerRoot '.venv\Scripts\pythonw.exe') -Argument ('"' + (Join-Path $workerRoot 'worker.py') + '"') -WorkingDirectory $workerRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userName
$principal = New-ScheduledTaskPrincipal -UserId $userName -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'VideoTranslator Home Downloader' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
$stopPath = Join-Path $workerRoot 'STOP'
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
Start-ScheduledTask -TaskName 'VideoTranslator Home Downloader'
Write-Host 'Enabled: starts silently at login under your user account. It does not run before login.'
