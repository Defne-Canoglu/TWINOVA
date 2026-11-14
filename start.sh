#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python -m v136.release --demo
