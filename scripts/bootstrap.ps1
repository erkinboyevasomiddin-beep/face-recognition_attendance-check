[CmdletBinding()]
param(
    [switch]$Recognition,
    [switch]$SyntheticData,
    [string]$BasePython = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    if ($BasePython) {
        & $BasePython -m venv $VenvPath
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv $VenvPath
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $VenvPath
    }
    else {
        throw "Python 3.11-3.13 was not found. Install Python or pass -BasePython <path>."
    }
}

& $PythonPath -m pip install "pip>=26.1.2"
& $PythonPath -m pip install -e "$ProjectRoot[dev]"
if ($Recognition) {
    & $PythonPath -m pip install -e "$ProjectRoot[recognition]"
}

$EnvPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvPath)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvPath
    Write-Warning "Created .env from placeholders. Generate unique secrets before enabling the API."
}

if ($SyntheticData) {
    Push-Location $ProjectRoot
    try {
        & $PythonPath -m backend.scripts.setup_demo
    }
    finally {
        Pop-Location
    }
}
