#!/bin/bash
# Build the Docker image and push to the cluster's internal registry.
# Run this ON the cluster login node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${1:-posttrainbench:latest}"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "${SCRIPT_DIR}/../Dockerfile" "${SCRIPT_DIR}/.."

echo "Done. Image ready: $IMAGE_NAME"
echo ""
echo "To run a test job:"
echo "  mkdir -p logs"
echo "  sbatch ${SCRIPT_DIR}/submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 1 claude-opus-4-6"
