$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Ensure-Tool($Command, $Package) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "Install $Package first (winget / App Installer is unavailable). See README.md."
        }
        & winget install --id $Package --exact --source winget --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw "Could not install $Package" }
        Refresh-Path
    }
}

Write-Host 'Setting up the private VideoTranslator download worker (no inbound ports).'
Ensure-Tool 'py' 'Python.Python.3.12'
& py -3.12 -c 'import sys; assert sys.version_info >= (3,12)'
if ($LASTEXITCODE -ne 0) {
    & winget install --id Python.Python.3.12 --exact --source winget --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 installation failed.' }
    Refresh-Path
}
Ensure-Tool 'node' 'OpenJS.NodeJS.LTS'
Ensure-Tool 'ffmpeg' 'Gyan.FFmpeg'
& py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

$configPath = Join-Path $PSScriptRoot 'config.json'
if (-not (Test-Path -LiteralPath $configPath)) {
    $serverUrl = Read-Host 'Server URL [https://vidyi.cc]'
    if (-not $serverUrl) { $serverUrl = 'https://vidyi.cc' }
    $secureToken = Read-Host 'Private worker token from the server administrator' -AsSecureString
    $tokenValue = [System.Net.NetworkCredential]::new('', $secureToken).Password
    if ($tokenValue.Length -lt 32) { throw 'The token must have at least 32 characters.' }
    @{server_url=$serverUrl;token=$tokenValue;node_path=(Get-Command node).Source;ffmpeg_path=(Get-Command ffmpeg).Source} |
        ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
    $tokenValue = $null
}
$workerConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$workerConfig.node_path = (Get-Command node).Source
$workerConfig.ffmpeg_path = (Get-Command ffmpeg).Source
$workerConfig | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
# Restrict credentials to this Windows user and SYSTEM.
$acl = New-Object System.Security.AccessControl.FileSecurity
$acl.SetAccessRuleProtection($true, $false)
$ownerSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
foreach ($sid in @($ownerSid, [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18'))) {
    $acl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', 'Allow'))
}
Set-Acl -LiteralPath $configPath -AclObject $acl
& .\.venv\Scripts\python.exe worker.py --check
if ($LASTEXITCODE -ne 0) { throw 'Invalid configuration; check config.json.' }
Write-Host 'Ready. Double-click Start.cmd to run, or run Enable-Autostart.ps1 to start at Windows login.'
Write-Host 'Disable Windows sleep while this computer is serving downloads. Logs: worker.log'
