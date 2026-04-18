#!/usr/bin/env bash
# Runs cargo flamegraph on the world-sim diagnostic.
#
# Usage: scripts/perf_bench.sh [ticks] [world] [output_dir]
# Defaults: 500 ticks, default world, generated/flamegraphs

set -euo pipefail

TICKS="${1:-500}"
WORLD="${2:-}"           # "" → default world, "small" → small world
OUT_DIR="${3:-generated/flamegraphs}"

mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="$([ -z "$WORLD" ] && echo "default" || echo "$WORLD")"
SVG="$OUT_DIR/world_sim_${TAG}_${TS}.svg"

WORLD_FLAG=""
[ -n "$WORLD" ] && WORLD_FLAG="--world $WORLD"

echo "flamegraph: ticks=$TICKS world=${TAG} → $SVG"

# cargo-flamegraph's --root will use sudo for perf if needed.
# Running without --features profile-systems to avoid instrumentation overhead;
# perf samples the stack at 99Hz, giving intra-function hot lines for free.
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo flamegraph \
  --bin xtask \
  --release \
  --output "$SVG" \
  -- world-sim --ticks "$TICKS" $WORLD_FLAG

echo ""
echo "flamegraph: $SVG"
echo "open: file://$(realpath "$SVG")"
