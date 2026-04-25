#!/bin/bash
set -e

die() {
    echo "Error: $1" >&2
    exit 1
}

command -v uv &>/dev/null || die "'uv' is required. Install from https://docs.astral.sh/uv/getting-started/installation/"
command -v git &>/dev/null || die "'git' is required. Please install git and try again."

REPO_URL="https://github.com/brains-group/table-agent-toolkit.git"
TMP_DIR="$(mktemp -d)"

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "Cloning table-agent-toolkit..."
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

echo "Installing..."
uv tool install "$TMP_DIR"

table-agent-toolkit-install
