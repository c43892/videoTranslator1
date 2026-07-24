param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [string]$TargetLanguage = "English",
    [string]$SubscriptionId = "65168226-66f0-4c42-ba76-6dfde11c451e",
    [string]$Location = "eastus",
    [string]$OutputDirectory = ".\azure-output",
    [switch]$KeepAzureResources
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$environmentFile = Join-Path $PSScriptRoot "environment.yml"
$jobFile = Join-Path $PSScriptRoot "job.yml"
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$resourceGroup = "rg-videotranslator-$timestamp"
$workspace = "mlw-videotranslator-$timestamp"
$createdResourceGroup = $false

if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) {
    throw "Input video not found: $InputVideo"
}
$inputVideoPath = (Resolve-Path -LiteralPath $InputVideo).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

try {
    az account set --subscription $SubscriptionId

    & (Join-Path $PSScriptRoot "check-t4-quota.ps1") `
        -SubscriptionId $SubscriptionId `
        -Location $Location
    if ($LASTEXITCODE -ne 0) {
        throw "T4 quota is not ready. No Azure resources were created."
    }

    az group create --name $resourceGroup --location $Location --output none
    if ($LASTEXITCODE -ne 0) { throw "Resource group creation failed." }
    $createdResourceGroup = $true

    az ml workspace create `
        --name $workspace `
        --resource-group $resourceGroup `
        --location $Location `
        --output none
    if ($LASTEXITCODE -ne 0) { throw "Azure ML workspace creation failed." }

    az ml environment create `
        --file $environmentFile `
        --resource-group $resourceGroup `
        --workspace-name $workspace `
        --output none
    if ($LASTEXITCODE -ne 0) { throw "Azure ML environment registration failed." }

    Push-Location $repoRoot
    try {
        $jobName = az ml job create `
            --file $jobFile `
            --resource-group $resourceGroup `
            --workspace-name $workspace `
            --set `
                inputs.input_video.path=$inputVideoPath `
                inputs.target_language=$TargetLanguage `
            --query name `
            --output tsv
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0 -or -not $jobName) { throw "Azure ML job submission failed." }

    Write-Host "Azure ML job: $jobName"
    az ml job stream `
        --name $jobName `
        --resource-group $resourceGroup `
        --workspace-name $workspace
    if ($LASTEXITCODE -ne 0) { throw "Azure ML job failed." }

    az ml job download `
        --name $jobName `
        --download-path $outputPath `
        --resource-group $resourceGroup `
        --workspace-name $workspace
    if ($LASTEXITCODE -ne 0) { throw "Output download failed." }

    Write-Host "Output downloaded to $outputPath"
}
finally {
    if ($createdResourceGroup -and -not $KeepAzureResources) {
        Write-Host "Deleting temporary Azure resource group $resourceGroup..."
        az group delete --name $resourceGroup --yes --output none
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Automatic cleanup failed. Delete resource group '$resourceGroup' manually to stop any remaining charges."
        }
    }
}
