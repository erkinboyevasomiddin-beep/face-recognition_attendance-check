[CmdletBinding()]
param([switch]$Apply)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$Targets = @(
    (Join-Path $ProjectRoot ".pytest_cache"),
    (Join-Path $ProjectRoot ".mypy_cache"),
    (Join-Path $ProjectRoot ".ruff_cache"),
    (Join-Path $ProjectRoot "htmlcov"),
    (Join-Path $ProjectRoot ".coverage")
)

foreach ($Target in $Targets) {
    $ResolvedParent = (Resolve-Path (Split-Path -Parent $Target) -ErrorAction SilentlyContinue)
    if ($null -eq $ResolvedParent -or -not $ResolvedParent.Path.StartsWith($ProjectRoot)) {
        throw "Refusing to process a target outside the project: $Target"
    }
    if (Test-Path -LiteralPath $Target) {
        if ($Apply) {
            Remove-Item -LiteralPath $Target -Recurse -Force
            Write-Host "Removed $Target"
        }
        else {
            Write-Host "Would remove $Target"
        }
    }
}

if (-not $Apply) {
    Write-Host "Preview only. Re-run with -Apply after reviewing the exact targets."
}
