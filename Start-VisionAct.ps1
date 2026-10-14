$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirements = Join-Path $projectRoot "requirements.txt"
$appDataRoot = Join-Path $env:LOCALAPPDATA "VisionAct"
$environment = Join-Path $appDataRoot "venv"
$python = Join-Path $environment "Scripts\python.exe"
$marker = Join-Path $appDataRoot "requirements.sha256"

try {
    New-Item -ItemType Directory -Path $appDataRoot -Force | Out-Null

    if (-not (Test-Path -LiteralPath $python)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) {
            throw "Python 3.10 or newer is required. Install Python from https://www.python.org/downloads/ and enable the Python Launcher, then try again."
        }
        & $launcher.Source -3 -m venv $environment
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the VisionAct Python environment."
        }
    }

    $pythonVersion = & $python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not start the VisionAct Python environment."
    }
    if ($pythonVersion -notmatch "Python 3\.(\d+)") {
        throw "Could not determine the Python version: $pythonVersion"
    }
    if ([int]$Matches[1] -lt 10) {
        throw "VisionAct requires Python 3.10 or newer. Found $pythonVersion."
    }

    $requirementsHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
    $installedHash = if (Test-Path -LiteralPath $marker) {
        (Get-Content -LiteralPath $marker -Raw).Trim()
    } else {
        ""
    }

    $needsInstall = $requirementsHash -ne $installedHash
    if (-not $needsInstall) {
        & $python -m streamlit --version *> $null
        $needsInstall = $LASTEXITCODE -ne 0
    }

    if ($needsInstall) {
        Write-Host "Setting up or updating VisionAct dependencies. This can take several minutes..."
        & $python -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency installation failed. Check your internet connection and try again."
        }
        Set-Content -LiteralPath $marker -Value $requirementsHash -NoNewline
    }

    Set-Location -LiteralPath $projectRoot
    Write-Host "Starting VisionAct. Your browser will open at http://localhost:8501."
    Write-Host "Keep this window open while using the app. Press Ctrl+C to stop it."
    & $python -m streamlit run app.py
    if ($LASTEXITCODE -ne 0) {
        throw "VisionAct stopped with an error."
    }
} catch {
    Write-Host ""
    Write-Host "VisionAct could not start: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
