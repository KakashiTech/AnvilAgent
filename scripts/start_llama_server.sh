#!/usr/bin/env bash
# Start llama-server with Llama-3.2-3B on port 8081 (Vulkan iGPU)
set -euo pipefail

MODEL="${HOME}/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_SERVER="${HOME}/llama.cpp/build/bin/llama-server"
PORT=${1:-8081}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Run scripts/anvil_setup.sh first"
    exit 1
fi

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    -c 16384 \
    -ngl 99 \
    --flash-attn \
    --no-mmap
