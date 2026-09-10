param(
    [string]$WeightsDir = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not $WeightsDir) {
    $WeightsDir = Join-Path $projectRoot 'weights'
}
New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null

$weightFile = Join-Path $WeightsDir 'KongNet_CoNIC_1.pth'
$expectedSha = '498ad0f328dcda96cd4d2314ebe40df1cc8d897af5a3b3e3732e6f78664c6af7'
$hfUrl = 'https://huggingface.co/TIACentre/KongNet_pretrained_weights/resolve/main/KongNet_CoNIC_1.pth'

if (Test-Path -LiteralPath $weightFile) {
    Write-Host "[weights] KongNet_CoNIC_1.pth already exists, verifying SHA256 ..."
    $actualSha = (Get-FileHash $weightFile -Algorithm SHA256).Hash.ToLower()
    if ($actualSha -eq $expectedSha) {
        Write-Host "[weights] SHA256 verified. Skipping download."
        return
    }
    Write-Host "[weights] SHA256 mismatch ($actualSha). Re-downloading ..."
    Remove-Item $weightFile
}

Write-Host "[weights] Downloading KongNet_CoNIC_1.pth from HuggingFace (~176 MB) ..."
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri $hfUrl -OutFile $weightFile -UseBasicParsing
$ProgressPreference = 'Continue'

$actualSha = (Get-FileHash $weightFile -Algorithm SHA256).Hash.ToLower()
if ($actualSha -ne $expectedSha) {
    throw "SHA256 mismatch after download! Expected $expectedSha, got $actualSha"
}
Write-Host "[weights] Download complete. SHA256 verified."
