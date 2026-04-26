#!/usr/bin/env bash
# Start the world sim with the voxel renderer.
#
# Usage:
#   ./game_start.sh                  # default: small world, rendered
#   ./game_start.sh --big             # infinite world preset
#   ./game_start.sh --peaceful        # peaceful mode (no monsters)
#   ./game_start.sh --seed 123        # override seed
#   ./game_start.sh -- <extra args>   # pass raw args to xtask world-sim
#
# Any unrecognized args are forwarded to `xtask world-sim` unchanged.
set -e
cd "$(dirname "$0")"

# Raise soft fd limit to the hard cap — Mesa/Vulkan driver allocates
# many sync fds during pipeline setup and has crashed with EMFILE on
# systems where the shell inherits a low soft limit.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo ">> ulimit -n: $(ulimit -n) (hard: $(ulimit -Hn))"
echo ">> system fd use: $(awk '{print $1"/"$3}' /proc/sys/fs/file-nr)"

WORLD="small"
SEED="42"
EXTRA=()
PEACEFUL=0
RICH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --small)    WORLD="small"; shift ;;
        --big)      WORLD=""; shift ;;
        --peaceful) PEACEFUL=1; shift ;;
        --rich)     RICH=1; shift ;;
        --seed)     SEED="$2"; shift 2 ;;
        --)         shift; EXTRA+=("$@"); break ;;
        *)          EXTRA+=("$1"); shift ;;
    esac
done

ARGS=(--render --seed "$SEED")
[[ -n "$WORLD" ]] && ARGS+=(--world "$WORLD")
[[ "$PEACEFUL" == "1" ]] && ARGS+=(--peaceful)
[[ "$RICH" == "1" ]] && ARGS+=(--rich)
ARGS+=("${EXTRA[@]}")

echo ">> cargo run --release --features app --bin xtask -- world-sim ${ARGS[*]}"
exec cargo run --release --features app --bin xtask -- world-sim "${ARGS[@]}"
