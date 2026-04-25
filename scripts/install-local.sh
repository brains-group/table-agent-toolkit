#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

die() { echo "Error: $1" >&2; exit 1; }

command -v uv &>/dev/null || die "'uv' is required. Install from https://docs.astral.sh/uv/getting-started/installation/"

echo "Installing table-agent-toolkit (editable)..."
uv tool install --editable "$PROJECT_DIR"

table-agent-toolkit-install
