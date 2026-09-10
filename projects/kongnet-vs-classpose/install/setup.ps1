<#
.SYNOPSIS
    一键安装：KongNet 与 ClassPose 病理核检测对比环境。
.DESCRIPTION
    串联执行：
    1. 初始化 git submodule（拉取 KongNet 和 ClassPose 作者源码）
    2. 下载 KongNet CoNIC 权重（HuggingFace, ~176 MB）
    3. 创建双 Python 虚拟环境（KongNet + ClassPose）并验证 CUDA
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install/setup.ps1
#>
param(
    [string]$Proxy = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $projectRoot

Push-Location $repoRoot
try {
    # ── Step 1: Git submodules ──
    Write-Host "`n=== [1/3] Initializing git submodules ===" -ForegroundColor Cyan
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }

    # Verify vendor dirs exist
    foreach ($vendor in @('KongNet_Inference_Main', 'classpose')) {
        $vendorPath = Join-Path $projectRoot "vendor/$vendor"
        if (-not (Test-Path $vendorPath)) {
            throw "Vendor directory missing after submodule init: $vendorPath"
        }
    }
    Write-Host "[ok] Submodules ready.`n"
} finally {
    Pop-Location
}

# ── Step 2: Download weights ──
Write-Host "=== [2/3] Downloading model weights ===" -ForegroundColor Cyan
& "$PSScriptRoot\download-weights.ps1"
Write-Host "[ok] Weights ready.`n"

# ── Step 3: Python environments ──
Write-Host "=== [3/3] Installing Python environments ===" -ForegroundColor Cyan
$envArgs = @()
if ($Proxy) { $envArgs += @('-Proxy', $Proxy) }
& "$PSScriptRoot\install-environments.ps1" @envArgs
Write-Host ""

# ── Done ──
Write-Host "============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run KongNet WSI inference:"
Write-Host "  .venv\Scripts\python.exe scripts\run_kongnet_wsi_windows.py <args>"
Write-Host ""
Write-Host "Run ClassPose WSI inference:"
Write-Host "  .venv-classpose\Scripts\python.exe scripts\run_classpose_wsi.py <args>"
Write-Host ""
Write-Host "See README.md for full usage examples."
