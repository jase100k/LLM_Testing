#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if command -v hf &>/dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD="huggingface-cli"
else
    echo "❌ Error: neither 'hf' nor 'huggingface-cli' found."
    exit 1
fi

MODEL_REPO="Qwen/Qwen-Image-2.1"
TARGET_DIR="${SCRIPT_DIR}/models/Qwen-Image-2.1"

echo "=========================================================================="
echo "  📥 Downloading ${MODEL_REPO} via ${HF_CMD} download"
echo "  Target: ${TARGET_DIR}"
echo "=========================================================================="

mkdir -p "$TARGET_DIR"

$HF_CMD download \
    "${MODEL_REPO}" \
    --local-dir "${TARGET_DIR}"

echo ""
echo "✅ Download complete! Model stored in: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
