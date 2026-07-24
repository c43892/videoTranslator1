param(
    [string]$SubscriptionId = "65168226-66f0-4c42-ba76-6dfde11c451e",
    [string]$Location = "eastus"
)

$ErrorActionPreference = "Stop"

az account set --subscription $SubscriptionId
$usage = az vm list-usage --location $Location --output json | ConvertFrom-Json
$t4 = $usage | Where-Object { $_.name.value -eq "Standard NCASv3_T4 Family" }

if (-not $t4) {
    throw "Azure did not return the NCASv3_T4 quota entry for $Location."
}

[pscustomobject]@{
    Subscription = $SubscriptionId
    Location     = $Location
    Family       = $t4.name.localizedValue
    UsedVCPUs    = $t4.currentValue
    LimitVCPUs   = $t4.limit
    ReadyForNC4  = ($t4.limit - $t4.currentValue) -ge 4
} | Format-List

if (($t4.limit - $t4.currentValue) -lt 4) {
    throw "T4 quota is not ready. At least four available vCPUs are required."
}
