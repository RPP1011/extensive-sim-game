#!/usr/bin/env bash
# Rebuild docs/superpowers/dag/state.json from ROADMAP + plans.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 .claude/skills/project-dag/scripts/bootstrap.py "$@"
