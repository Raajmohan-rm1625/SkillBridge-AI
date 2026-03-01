#!/bin/bash
# SkillBridge AI — AMD ROCm Setup & Deployment Script
# Tested on: AMD Instinct MI300X, ROCm 6.x, Ubuntu 22.04

set -e

echo "=== SkillBridge AI — AMD ROCm Setup ==="

# 1. Install ROCm 6.x
echo "[1/6] Installing ROCm..."
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install -y --usecase=rocm
sudo usermod -a -G render,video $LOGNAME

# 2. Verify GPU
echo "[2/6] Verifying AMD GPU..."
rocm-smi
rocminfo | grep "Device Type" | head -5

# 3. Python environment
echo "[3/6] Setting up Python environment..."
python3 -m venv skillbridge_env
source skillbridge_env/bin/activate

# 4. Install PyTorch for ROCm
echo "[4/6] Installing PyTorch (ROCm 6.0 build)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

# Verify PyTorch sees AMD GPU
python3 -c "import torch; print(f'CUDA (ROCm) available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 5. Install vLLM (ROCm fork for optimized LLM serving)
echo "[5/6] Installing vLLM for AMD..."
pip install vllm  # ROCm auto-detected at build time
# Alternative: pip install "vllm[rocm]"

# 6. Install app dependencies
echo "[6/6] Installing SkillBridge AI dependencies..."
pip install \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    transformers==4.41.0 \
    accelerate==0.30.0 \
    sentence-transformers==3.0.0 \
    langchain==0.2.0 \
    langchain-community==0.2.0 \
    pinecone-client==3.2.2 \
    psycopg2-binary==2.9.9 \
    redis==5.0.4 \
    pydantic==2.7.1 \
    python-multipart==0.0.9

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start SkillBridge AI backend:"
echo "  source skillbridge_env/bin/activate"
echo "  python main.py"
echo ""
echo "AMD MI300X Inference tips:"
echo "  - Set HIP_VISIBLE_DEVICES=0 to target specific GPU"
echo "  - Use bfloat16 dtype for best MI300X performance"  
echo "  - Enable torch.compile() for 20-30% speedup on ROCm"
echo "  - vLLM tensor_parallel_size=2+ for multi-GPU MI300X"
