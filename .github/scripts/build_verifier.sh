#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Reproduce the committed verifier on a clean Linux Docker host (09 §9).
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'usage: build_verifier.sh TARGET OUTPUT_DIRECTORY' >&2
  exit 2
fi
case "$1" in
  x86_64-unknown-linux-gnu) platform=linux/amd64 ;;
  aarch64-unknown-linux-gnu) platform=linux/arm64 ;;
  *) echo 'unsupported release target' >&2; exit 2 ;;
esac
repo=$(git rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse HEAD)
epoch=$(git -C "$repo" show -s --format=%ct HEAD)
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo 'release builds require a clean checkout of the release commit' >&2
  exit 1
fi
mkdir -p "$2"
output=$(cd "$2" && pwd)
scratch=$(mktemp -d)
trap 'rmdir "$scratch" 2>/dev/null || true' EXIT
# Archive only committed files; no local Cargo configuration or build cache enters the build.
git -C "$repo" archive HEAD > "$scratch/source.tar"
image=rust:1.95.0-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1
docker run --rm -i --platform "$platform" --network bridge \
  --mount "type=bind,src=$scratch/source.tar,dst=/source.tar,readonly" \
  --mount "type=bind,src=$output,dst=/out" \
  --env "SOURCE_DATE_EPOCH=$epoch" --env "TESSERA_GIT_COMMIT=$commit" \
  --env "RELEASE_TARGET=$1" --env "RELEASE_IMAGE=$image" \
  "$image" bash -euo pipefail -s <<'BUILD'
mkdir /source /build /cargo
tar -xf /source.tar -C /source
cd /source/verifier
channel=$(sed -nE 's/^channel *= *"([^"]+)".*/\1/p' rust-toolchain.toml)
test "$(rustup show active-toolchain | cut -d' ' -f1)" = "$channel-$RELEASE_TARGET"
export CARGO_HOME=/cargo CARGO_TARGET_DIR=/build CARGO_INCREMENTAL=0
export TESSERA_RUSTC_VERSION="$(rustc --version)"
export RUSTFLAGS='--remap-path-prefix=/source=/tessera --remap-path-prefix=/cargo=/cargo-home --remap-path-prefix=/build=/target -C target-cpu=generic'
export TZ=UTC LC_ALL=C
umask 022
cargo build --release --locked --target "$RELEASE_TARGET" -p tv-cli --bin tessera-verify
install -m 755 "/build/$RELEASE_TARGET/release/tessera-verify" "/out/tessera-verify-$RELEASE_TARGET"
# No host paths, timestamps of the run or runner names enter the reproducibility record.
printf '{"commit":"%s","epoch":%s,"target":"%s","image":"%s","rustc":"%s"}\n' \
  "$TESSERA_GIT_COMMIT" "$SOURCE_DATE_EPOCH" "$RELEASE_TARGET" "$RELEASE_IMAGE" "$TESSERA_RUSTC_VERSION" > /out/build.json
BUILD
# Remove only the scratch archive this script created.
unlink "$scratch/source.tar"
