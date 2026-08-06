[CmdletBinding()]
param([switch]$Fix)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VersionCheckPath = Join-Path $PSScriptRoot "check_python_version.py"
if (-not (Test-Path $PythonPath)) {
    throw "Run scripts\bootstrap.ps1 first."
}

& $PythonPath $VersionCheckPath
if ($LASTEXITCODE -ne 0) {
    throw "Quality checks require Python 3.11."
}

function Invoke-PythonCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $ProjectRoot
try {
    if ($Fix) {
        Invoke-PythonCommand @("-m", "ruff", "check", ".", "--fix")
        Invoke-PythonCommand @("-m", "ruff", "format", ".")
    }
    else {
        Invoke-PythonCommand @("-m", "ruff", "check", ".")
        Invoke-PythonCommand @("-m", "ruff", "format", "--check", ".")
    }
    Invoke-PythonCommand @("-m", "mypy", "backend", "recognition")
    Invoke-PythonCommand @("-m", "pytest")
    Invoke-PythonCommand @("-m", "pip", "check")
    Invoke-PythonCommand @(
        "-m",
        "pip_audit",
        "--cache-dir",
        (Join-Path $ProjectRoot ".local\pip-audit-cache"),
        "--skip-editable"
    )
}
finally {
    Pop-Location
}
