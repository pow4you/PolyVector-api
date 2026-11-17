#!/usr/bin/env bash
# Install the C++ dependencies of polyvec._native with vcpkg (Linux / macOS).
#
#   ci/install-vcpkg-deps.sh <triplet> [vcpkg-root] [install-root]
#
# Clones vcpkg at the commit pinned in src/polyvec/native/vcpkg.json (unless
# vcpkg-root already holds a checkout), bootstraps it, and installs the
# manifest's dependencies as static libraries for <triplet> (one of the
# overlay triplets in src/polyvec/native/vcpkg-triplets).  Afterwards, build
# with the environment printed at the end.  Used by cibuildwheel's before-all
# step; also works by hand.
set -euo pipefail

triplet="${1:?usage: $0 <triplet> [vcpkg-root] [install-root]}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
native="$here/src/polyvec/native"
root="${2:-${VCPKG_ROOT:-$HOME/vcpkg}}"
installed="${3:-$root/installed-polyvec}"
baseline="$(sed -n 's/.*"builtin-baseline": *"\([0-9a-f]*\)".*/\1/p' "$native/vcpkg.json")"

if [ ! -x "$root/vcpkg" ]; then
    if [ ! -d "$root/.git" ]; then
        git clone --filter=tree:0 https://github.com/microsoft/vcpkg.git "$root"
    fi
    git -C "$root" checkout --quiet "$baseline"
    "$root/bootstrap-vcpkg.sh" -disableMetrics
fi

if [ -n "${VCPKG_DEFAULT_BINARY_CACHE:-}" ]; then
    mkdir -p "$VCPKG_DEFAULT_BINARY_CACHE"   # vcpkg refuses a missing cache directory
fi

"$root/vcpkg" install \
    --x-manifest-root="$native" \
    --x-install-root="$installed" \
    --overlay-triplets="$native/vcpkg-triplets" \
    --triplet="$triplet" \
    --clean-after-build

cat <<INFO

vcpkg dependencies for $triplet are in $installed.  Build polyvec with:
  CMAKE_TOOLCHAIN_FILE=$root/scripts/buildsystems/vcpkg.cmake
  SKBUILD_CMAKE_DEFINE="VCPKG_TARGET_TRIPLET=$triplet;VCPKG_INSTALLED_DIR=$installed;VCPKG_MANIFEST_INSTALL=OFF;VCPKG_OVERLAY_TRIPLETS=$native/vcpkg-triplets"
INFO
