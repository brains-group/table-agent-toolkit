#!/bin/bash
set -e

die() {
    echo "Error: $1" >&2
    exit 1
}

command -v uv &>/dev/null || die "'uv' is required. Install from https://docs.astral.sh/uv/getting-started/installation/"

REPO_URL="https://github.com/inwonakng/table-agent-toolkit.git"
TMP_DIR="$(mktemp -d)"

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "Installing..."
uv tool install "$TMP_DIR"

table-agent-toolkit-install
