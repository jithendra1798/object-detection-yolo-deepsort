#!/usr/bin/env bash
set -euo pipefail

echo "Building Docker image..."
docker build -t object-tracker:local .

echo "Running container (press Ctrl+C to stop)..."
docker run --rm -p 5000:5000 -v "$(pwd)/uploads:/app/uploads" -v "$(pwd)/processed:/app/processed" --name object-tracker-local object-tracker:local
