param([switch]$DownloadWeights)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
Get-Command uv, git -ErrorAction Stop | Out-Null
Invoke-Checked uv @('venv', '--python', '3.12', 'repo/.venv')
$vendorPath = Join-Path $projectRoot 'repo/vendor/nnUNet'
if (!(Test-Path -LiteralPath $vendorPath)) {
    Invoke-Checked git @('clone', 'https://github.com/MIC-DKFZ/nnUNet.git', $vendorPath)
}
$vendorChanges = & git -C $vendorPath status --porcelain
if ($LASTEXITCODE -ne 0 -or $vendorChanges) { throw 'nnUNet checkout is not clean; preserve it and use a fresh project directory.' }
Invoke-Checked git @('-C', $vendorPath, 'checkout', '--detach', '202f6baa0adc2ef5f7b615df19cc4da970412cc0')
$pythonPath = Join-Path $projectRoot 'repo/.venv/Scripts/python.exe'
Invoke-Checked uv @('pip', 'install', '--python', $pythonPath, '-r', 'repo/requirements-lock.txt', '--extra-index-url', 'https://download.pytorch.org/whl/cu126', '--index-strategy', 'unsafe-best-match')
Invoke-Checked $pythonPath @('-c', 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name())')
if ($DownloadWeights) { Invoke-Checked $pythonPath @('repo/fetch_weights.py', '--download') }
Write-Host 'Environment ready. Follow README for the TumSeg archive and frozen experiment sequence.'
