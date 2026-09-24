$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$keyPath = Join-Path $project 'secrets/azure-jp-ed25519'
if (-not (Test-Path -LiteralPath $keyPath)) {
    ssh-keygen -t ed25519 -f $keyPath -N '' -C 'videotranslator-azure-validation'
    if ($LASTEXITCODE -ne 0) { throw 'SSH key generation failed' }
    icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(F)" | Out-Null
}
$publicIp = (Invoke-RestMethod -Uri 'https://api.ipify.org' -TimeoutSec 20).Trim()
$parsedIp = [System.Net.IPAddress]::Parse($publicIp)
if ($parsedIp.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) { throw 'An IPv4 management address is required' }
$group = 'videotranslator-jpe-rg'
function Invoke-Az {
    & az @args --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "Azure operation failed: $($args[0..1] -join ' ')" }
}
Invoke-Az network nsg create -g $group -n videotranslator-cpu-nsg -l japaneast -o none
Invoke-Az network nsg rule create -g $group --nsg-name videotranslator-cpu-nsg -n ssh-admin --priority 100 --source-address-prefixes "$publicIp/32" --destination-port-ranges 22 --access Allow --protocol Tcp -o none
Invoke-Az network nsg rule create -g $group --nsg-name videotranslator-cpu-nsg -n web --priority 110 --source-address-prefixes Internet --destination-port-ranges 80 443 --access Allow --protocol Tcp -o none
Invoke-Az network nsg rule create -g $group --nsg-name videotranslator-cpu-nsg -n aca-postgres --priority 120 --source-address-prefixes 10.0.0.0/23 --destination-port-ranges 5432 --access Allow --protocol Tcp -o none
Invoke-Az network nsg rule create -g $group --nsg-name videotranslator-cpu-nsg -n deny-other-inbound --priority 4000 --source-address-prefixes '*' --destination-port-ranges '*' --access Deny --protocol '*' -o none
Invoke-Az network vnet subnet create -g $group --vnet-name videotranslator-jpe-vnet -n cpu --address-prefixes 10.0.2.0/24 --network-security-group videotranslator-cpu-nsg -o none
Invoke-Az vm create -g $group -n videotranslator-cpu -l japaneast --image Ubuntu2404 --size Standard_D2as_v4 --admin-username vtadmin --ssh-key-values "$keyPath.pub" --vnet-name videotranslator-jpe-vnet --subnet cpu --nsg videotranslator-cpu-nsg --public-ip-sku Standard --public-ip-address-dns-name vtranslator-jpe-43892 --storage-sku StandardSSD_LRS --os-disk-size-gb 64 --assign-identity --custom-data "$PSScriptRoot/cpu-cloud-init.yml" --tags purpose=videotranslator-sandbox validationBudgetUSD=20 -o json
$shutdown = [DateTime]::UtcNow.AddHours(8).ToString('HHmm')
Invoke-Az vm auto-shutdown -g $group -n videotranslator-cpu --time $shutdown -o none
Write-Output "Validation VM automatic shutdown configured for $shutdown UTC."
