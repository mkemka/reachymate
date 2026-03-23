# Install Reachy Convo Mate into the Reachy Mini Control desktop app's Python environment
# so it appears in the app list. Run from repo root or pass -ReachymatePath.

param(
    [string]$ReachymatePath = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path $ReachymatePath).Path
if (-not (Test-Path (Join-Path $repoRoot "pyproject.toml"))) {
    Write-Error "Reachymate repo not found at: $repoRoot (no pyproject.toml)"
}

# Possible locations for the desktop app's bundle (where uv-trampoline and Python live)
$searchPaths = @(
    "$env:LOCALAPPDATA\Programs\Reachy Mini Control",
    "$env:LOCALAPPDATA\Reachy Mini Control",
    "$env:APPDATA\Reachy Mini Control",
    "${env:ProgramFiles}\Reachy Mini Control",
    "${env:ProgramFiles(x86)}\Reachy Mini Control"
)

# Also search in the same drive for uv-trampoline
$drives = @($env:SystemDrive + "\")
try {
    $drives += (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root } | ForEach-Object { $_.Root })
} catch {}

function Find-BundlePython {
    $found = $null
    foreach ($base in $searchPaths) {
        if (-not (Test-Path $base)) { continue }
        $binaries = Join-Path $base "binaries"
        if (Test-Path $binaries) {
            # uv-bundle may create python.exe in binaries or in a venv
            $pythonExe = Join-Path $binaries "python.exe"
            if (Test-Path $pythonExe) { return $pythonExe }
            $venvPython = Join-Path $binaries ".venv\Scripts\python.exe"
            if (Test-Path $venvPython) { return $venvPython }
            $scriptsPython = Join-Path $binaries "Scripts\python.exe"
            if (Test-Path $scriptsPython) { return $scriptsPython }
        }
        # Some bundles put python next to the exe
        $pythonExe = Join-Path $base "python.exe"
        if (Test-Path $pythonExe) { return $pythonExe }
    }
    # Fallback: search for uv-trampoline and assume Python is in same dir or parent
    $trampoline = Get-Command uv-trampoline -ErrorAction SilentlyContinue
    if ($trampoline) {
        $dir = Split-Path $trampoline.Source
        $pythonExe = Join-Path $dir "python.exe"
        if (Test-Path $pythonExe) { return $pythonExe }
    }
    return $null
}

$bundlePython = Find-BundlePython
if (-not $bundlePython) {
    Write-Host "Could not find the Reachy Mini Control desktop app's Python environment."
    Write-Host "Searched in: $($searchPaths -join ', ')"
    Write-Host ""
    Write-Host "If you installed the app elsewhere:"
    Write-Host "  1. Find the app install folder (e.g. where 'Reachy Mini Control.exe' or 'uv-trampoline.exe' is)."
    Write-Host "  2. Look for a 'binaries' subfolder containing 'python.exe' or '.venv\Scripts\python.exe'."
    Write-Host "  3. Run manually (replace PATH_TO_PYTHON and PATH_TO_REACHYMATE):"
    Write-Host '     & "PATH_TO_PYTHON" -m pip install -e "PATH_TO_REACHYMATE"'
    Write-Host ""
    Write-Host "Then restart the Reachy Mini Control desktop app so it rescans installed apps."
    exit 1
}

Write-Host "Using Python: $bundlePython"
Write-Host "Installing Reachy Convo Mate (editable) from: $repoRoot"
& $bundlePython -m pip install -e $repoRoot
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed."
}
Write-Host "Done. Restart the Reachy Mini Control desktop app so 'Reachy Convo Mate' appears in the app list."
