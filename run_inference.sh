#!/bin/bash

# ==============================================================================
# CropRL Inference Setup & Run Script
# ==============================================================================
# This script automates the installation of:
# 1. uv (Python package manager)
# 2. Project dependencies (via uv)
# 3. Ollama server
# 4. Qwen 3.5 9B model
# Finally, it runs the inference script.
# ==============================================================================

set -e # Exit on error

echo "🚀 Starting setup for CropRL Inference..."

# 1. Install uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to current path for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    # Also add to .bashrc for future sessions if it's a fresh VM
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
else
    echo "✅ uv is already installed."
fi

# 2. Create and activate virtual environment
echo "🌱 Setting up Python virtual environment..."
uv venv .venv
source .venv/bin/activate

# 3. Install dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing base requirements..."
    uv pip install openenv-core numpy pydantic fastapi uvicorn openai tqdm pytest httpx
fi
echo "📥 Installing ollama python client..."
uv pip install ollama

# 4. Install Ollama Server
if ! command -v ollama &> /dev/null; then
    echo "🦙 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama is already installed."
fi

# 5. Run Ollama in background
echo "⚡ Starting Ollama server in background..."
# Kill any existing ollama serve process to avoid conflicts
pkill ollama || true
# Ensure the log file directory exists
mkdir -p logs
ollama serve > logs/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama server to respond..."
MAX_RETRIES=30
COUNT=0
until curl -s http://localhost:11434/api/tags > /dev/null || [ $COUNT -eq $MAX_RETRIES ]; do
    sleep 2
    ((COUNT++))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Ollama server failed to start in time. Check logs/ollama.log"
    exit 1
fi
echo "✅ Ollama server is up and running."

# 6. Download model
echo "📥 Pulling qwen3.5:9b model (this may take a few minutes)..."
ollama pull qwen3.5:9b

# 7. Run inference
echo "🚜 Running CropRL inference..."
export MODEL_NAME="qwen3.5:9b"
# Ensure we are in the root of the repo to avoid path issues
python3 cropRL/inference.py

echo "✅ Inference complete."
