#!/usr/bin/env bash
# Sync this workspace to a Linux CUDA host and run a command there in the pixi environment.
#
# usage: scripts/remote.sh <ssh-host> <remote-dir> <command...>
# example: scripts/remote.sh gpu-box dev/siphon cargo test -p siphon-cuda
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,5p' "$0" >&2
    exit 2
fi
host=$1 dir=$2
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)

rsync -a --delete \
    --exclude /target --exclude .pixi --exclude /.git --exclude /reference/.venv \
    --exclude /reference/out --exclude /quickbench/.venv --exclude /experiments --exclude /releases --exclude /.claude \
    "$root/" "$host:$dir/"
ssh -o BatchMode=yes "$host" "cd $(printf %q "$dir") && pixi run $(printf '%q ' "$@")"
