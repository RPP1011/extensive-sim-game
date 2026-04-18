#!/usr/bin/env bash
# Wraps the xtask world-sim diagnostic benchmark under cargo flamegraph.
#
# Usage: scripts/perf_bench.sh [ticks] [world] [output_dir]
# Defaults: 2000 ticks, default (infinite) world, generated/flamegraphs

set -euo pipefail

TICKS="${1:-2000}"
WORLD="${2:-}"           # "" → default world, "small" → small world
OUT_DIR="${3:-generated/flamegraphs}"

mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="$([ -z "$WORLD" ] && echo "default" || echo "$WORLD")"
SVG="$OUT_DIR/world_sim_${TAG}_${TS}.svg"

WORLD_FLAG=""
[ -n "$WORLD" ] && WORLD_FLAG="--world $WORLD"

echo "running flamegraph: ticks=$TICKS world=${TAG} → $SVG"

# Linux perf permission check. cargo-flamegraph will escalate via sudo if
# paranoid > 1; otherwise it runs as-is. We let the user handle sudo prompts.
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 3)"
if [ "$PARANOID" -gt 1 ]; then
  echo "(perf_event_paranoid=$PARANOID; cargo flamegraph will request sudo if needed)"
fi

cargo flamegraph \
  --bin xtask \
  --features profile-systems \
  --release \
  --output "$SVG" \
  -- world-sim --ticks "$TICKS" $WORLD_FLAG \
       --bench-json "${OUT_DIR}/world_sim_${TAG}_${TS}.json"

echo ""
echo "flamegraph: $SVG"
echo "bench json: ${OUT_DIR}/world_sim_${TAG}_${TS}.json"
