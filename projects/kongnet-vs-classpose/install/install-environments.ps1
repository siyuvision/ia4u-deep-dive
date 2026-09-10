param(
    [string]$Uv = "$env:USERPROFILE\.local\bin\uv.exe",
    [string]$Proxy = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($Proxy) { $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy }

function Run-Uv {
    & $Uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}

Write-Host "[env] Installing KongNet environment (.venv, Python 3.11) ..."
if (-not (Test-Path "$projectRoot/.venv/Scripts/python.exe")) {
    Run-Uv venv "$projectRoot/.venv" --python 3.11
}
Run-Uv pip install --python "$projectRoot/.venv/Scripts/python.exe" `
    --only-binary stringzilla `
    -r "$projectRoot/vendor/KongNet_Inference_Main/requirements.txt"
Run-Uv pip install --python "$projectRoot/.venv/Scripts/python.exe" `
    --torch-backend cu124 `
    'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' openslide-bin

Write-Host "[env] Installing ClassPose environment (.venv-classpose, Python 3.13) ..."
if (-not (Test-Path "$projectRoot/.venv-classpose/Scripts/python.exe")) {
    Run-Uv venv "$projectRoot/.venv-classpose" --python 3.13
}
Run-Uv pip install --python "$projectRoot/.venv-classpose/Scripts/python.exe" `
    --torch-backend cu126 -e "$projectRoot/vendor/classpose"
Run-Uv pip install --python "$projectRoot/.venv-classpose/Scripts/python.exe" `
    'cellpose==4.0.8'

Write-Host "[env] Validating CUDA and dependencies ..."
foreach ($envName in @('.venv', '.venv-classpose')) {
    & "$projectRoot/$envName/Scripts/python.exe" -c `
        "import torch, openslide; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
    if ($LASTEXITCODE -ne 0) { throw "Environment validation failed: $envName" }
}
Write-Host "[env] Both environments ready."
