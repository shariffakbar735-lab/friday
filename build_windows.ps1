$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $project

$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project .venv is missing. Create it with: py -3.12 -m venv .venv'
}
$spec = Join-Path $project 'packaging\FRIDAY.spec'
if (-not (Test-Path -LiteralPath $spec -PathType Leaf)) { throw "Missing build spec: $spec" }
$entry = Join-Path $project 'app\main.py'
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw "Missing FRIDAY entry point: $entry" }
$dist = Join-Path $project 'dist\FRIDAY'
$exe = Join-Path $dist 'FRIDAY.exe'
$work = Join-Path $project 'build\pyinstaller_retry'
if (Test-Path -LiteralPath $dist) {
    throw "Distribution output already exists; refusing to overwrite: $dist"
}
if (Test-Path -LiteralPath $work) {
    throw "PyInstaller work directory already exists; refusing to overwrite: $work"
}
Write-Output "PyInstaller spec: $spec"
Write-Output "Entry point: $entry"
Write-Output "Distribution output: $dist"
Write-Output "Expected executable: $exe"
Write-Output "PyInstaller work directory: $work"

& $python -m setup_friday --venv-dir (Join-Path $project '.venv') --requirements-file 'requirements-build.txt'
if ($LASTEXITCODE -ne 0) { throw "Build-tool installation failed with exit code $LASTEXITCODE" }

Push-Location -LiteralPath $project
try {
    & $python -m PyInstaller --noconfirm --clean --distpath (Join-Path $project 'dist') `
    --workpath $work $spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build did not produce $exe" }
$bytes = (Get-ChildItem -LiteralPath $dist -File -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Output "Build complete: $exe"
Write-Output ('One-folder payload size: {0:N1} MiB' -f ($bytes / 1MB))
Write-Output 'Ollama and model weights are not bundled.'
