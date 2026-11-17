# Install the C++ dependencies of polyvec._native with vcpkg (Windows).
#
#   ci\install-vcpkg-deps.ps1 [-Triplet x64-windows-polyvec] [-Root C:\vcpkg] [-Installed <dir>]
#
# Windows counterpart of install-vcpkg-deps.sh: clones vcpkg at the commit
# pinned in src\polyvec\native\vcpkg.json (unless -Root already holds a
# checkout), bootstraps it and installs the manifest's dependencies as static
# libraries linked against the dynamic MSVC runtime.  Needs Visual Studio (or
# its Build Tools) with the C++ workload.
param(
    [string]$Triplet = "x64-windows-polyvec",
    [string]$Root = $(if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "C:\vcpkg" }),
    [string]$Installed = ""
)
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $PSScriptRoot
$native = Join-Path $here "src\polyvec\native"
if (-not $Installed) { $Installed = Join-Path $Root "installed-polyvec" }
$manifest = Get-Content (Join-Path $native "vcpkg.json") -Raw | ConvertFrom-Json
$baseline = $manifest.'builtin-baseline'

if (-not (Test-Path (Join-Path $Root "vcpkg.exe"))) {
    if (-not (Test-Path (Join-Path $Root ".git"))) {
        git clone --filter=tree:0 https://github.com/microsoft/vcpkg.git $Root
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    git -C $Root checkout --quiet $baseline
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & (Join-Path $Root "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

if ($env:VCPKG_DEFAULT_BINARY_CACHE) {
    # vcpkg refuses a missing cache directory
    New-Item -ItemType Directory -Force $env:VCPKG_DEFAULT_BINARY_CACHE | Out-Null
}

& (Join-Path $Root "vcpkg.exe") install `
    "--x-manifest-root=$native" `
    "--x-install-root=$Installed" `
    "--overlay-triplets=$(Join-Path $native 'vcpkg-triplets')" `
    "--triplet=$Triplet" `
    --clean-after-build
if ($LASTEXITCODE) { exit $LASTEXITCODE }

$toolchain = Join-Path $Root "scripts\buildsystems\vcpkg.cmake"
$triplets = Join-Path $native "vcpkg-triplets"
Write-Host ""
Write-Host "vcpkg dependencies for $Triplet are in $Installed.  Build polyvec with:"
Write-Host "  `$env:CMAKE_TOOLCHAIN_FILE = '$toolchain'"
Write-Host "  `$env:SKBUILD_CMAKE_DEFINE = 'VCPKG_TARGET_TRIPLET=$Triplet;VCPKG_INSTALLED_DIR=$Installed;VCPKG_MANIFEST_INSTALL=OFF;VCPKG_OVERLAY_TRIPLETS=$triplets'"
