[CmdletBinding()]
param(
    [string]$Python,
    [switch]$SkipWeb
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Python) {
    $Candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $RepoRoot) ".venv\Scripts\python.exe")
    )
    $Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw "No project virtual-environment Python found. Create .venv or pass -Python explicitly."
}

& $Python -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Refusing to verify with a system/global Python environment."
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
Push-Location $RepoRoot
try {
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Python test suite failed." }

    & $Python -B -m pitalpha doctor
    if ($LASTEXITCODE -ne 0) { throw "Environment doctor failed." }

    & $Python -B -m pitalpha config validate configs\demo_synthetic.yaml
    if ($LASTEXITCODE -ne 0) { throw "Demo configuration validation failed." }

    if (-not $SkipWeb) {
        $WebRoot = Join-Path $RepoRoot "apps\web"
        if (Test-Path -LiteralPath (Join-Path $WebRoot "node_modules")) {
            Push-Location $WebRoot
            try {
                & npm.cmd run build
                if ($LASTEXITCODE -ne 0) { throw "Web production build failed." }
            }
            finally {
                Pop-Location
            }
        }
        else {
            Write-Host "Skipping web build: apps/web/node_modules is not installed."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "PIT Alpha Lab verification passed."
