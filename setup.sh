#!/bin/bash
# Script de Setup para QLoRA Training en RunPod
# Requisitos: Ubuntu 22.04 + CUDA 11.8+

set -e

echo "📦 Instalando dependencias del sistema (requiere sudo)..."
apt-get update
apt-get install -y python3-dev build-essential

echo "🚀 Instalando uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "📁 Creando entorno virtual..."
cd /workspace || cd ~
uv venv venv --python 3.10
source venv/bin/activate

echo "📦 Instalando dependencias críticas..."
# Instalación optimizada para RunPod/CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Unsloth y dependencias de entrenamiento
uv pip install unsloth
uv pip install trl peft accelerate bitsandbytes sentencepiece

# Dependencias adicionales para GGUF
uv pip install transformers datasets huggingface-hub

echo "📦 Instalando dependencias para Bonito dataset generation..."
# Bonito (BatsResearch) para generación sintética de datasets
uv pip install transformers torch accelerate

echo "✅ Setup completado!"
echo "Para activar el entorno: source venv/bin/activate"
echo "Para ejecutar el entrenamiento: python train.py"
