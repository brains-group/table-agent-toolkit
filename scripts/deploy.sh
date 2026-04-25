#!/bin/bash
set -e

if [ -f "$(dirname "$0")/../.env" ]; then
  export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
fi

if [ -z "$UV_PUBLISH_TOKEN" ]; then
  echo "Error: UV_PUBLISH_TOKEN is not set."
  exit 1
fi

echo "Building..."
rm -rf dist/
uv build

echo "Publishing..."
uv publish

echo "Done."
