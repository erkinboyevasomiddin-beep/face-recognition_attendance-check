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
$VersionCheckPath = Join-Path $PSScriptRoot "check_python_version.py"
$ConstraintsPath = Join-Path $ProjectRoot "constraints-python311.txt"

function Assert-Python311 {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$PrefixArguments = @()
    )

    & $Command @PrefixArguments $VersionCheckPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 is required."
    }
}

if (-not (Test-Path $PythonPath)) {
    if ($BasePython) {
        Assert-Python311 -Command $BasePython
        & $BasePython -m venv $VenvPath
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Assert-Python311 -Command "py" -PrefixArguments @("-3.11")
        py -3.11 -m venv $VenvPath
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Assert-Python311 -Command "python"
        python -m venv $VenvPath
    }
    else {
        throw "Python 3.11 was not found. Install it or pass -BasePython <path>."
    }
}

Assert-Python311 -Command $PythonPath
& $PythonPath -m pip install "pip>=26.1.2"
& $PythonPath -m pip install -c $ConstraintsPath -e "$ProjectRoot[dev]"
if ($Recognition) {
    & $PythonPath -m pip install -c $ConstraintsPath -e "$ProjectRoot[recognition]"
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
