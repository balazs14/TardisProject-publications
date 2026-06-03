$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$coreScript = Join-Path $scriptDir "bootstrap_publications_env.py"

if (-not (Test-Path -Path $coreScript)) {
    Write-Error "[PUB] error: core bootstrap script not found: $coreScript"
}

$pyCommand = $null
$pyArgs = @()

if ($env:PYTHON_BOOTSTRAP) {
    $pyCommand = $env:PYTHON_BOOTSTRAP
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCommand = "py"
    $pyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyCommand = "python"
} else {
    Write-Error "[PUB] error: no python interpreter found for bootstrap"
}

& $pyCommand @pyArgs $coreScript @args
exit $LASTEXITCODE
