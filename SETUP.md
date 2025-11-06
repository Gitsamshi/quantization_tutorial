# Setup Guide

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with CUDA support (4GB+ VRAM)
- RAM: 8GB system RAM
- Disk: 10GB free space

**Recommended:**
- GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)
- RAM: 16GB+ system RAM
- Disk: 20GB free space

**Optimal:**
- GPU: NVIDIA RTX 3090, A100, or better (24GB+ VRAM)
- RAM: 32GB+ system RAM
- Disk: 50GB free space

### Software Requirements

- Python 3.8 or higher
- CUDA 11.8+ or 12.0+
- pip or conda package manager

## Installation

### Option 1: Quick Install (pip)

```bash
# Clone the repository
git clone <repository-url>
cd quantization_tutorial

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Conda Install

```bash
# Create conda environment
conda create -n quantization python=3.10
conda activate quantization

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install transformers accelerate bitsandbytes matplotlib

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```
   If this fails, install/update your NVIDIA drivers.

2. **Reinstall PyTorch with CUDA:**
   ```bash
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA version compatibility:**
   ```bash
   nvcc --version  # Check installed CUDA version
   ```

### bitsandbytes Installation Issues

**Linux:**
```bash
pip install bitsandbytes
```

**Windows:**
bitsandbytes has limited Windows support. Try:
```bash
pip install bitsandbytes-windows
```

Or use WSL2 (Windows Subsystem for Linux).

**Mac:**
bitsandbytes requires CUDA, so it won't work on Mac. Examples 1, 2, and 5 will still work.

### auto-gptq Installation Issues

If auto-gptq installation fails:

```bash
# Make sure you have build tools installed
# Ubuntu/Debian:
sudo apt-get install build-essential

# Then try:
pip install auto-gptq --no-build-isolation
```

If it still fails, you can skip auto-gptq. Examples 1-3 and 5 will still work.

### Out of Memory Errors

If you encounter GPU OOM errors:

1. **Use a smaller model** in Example 3:
   ```python
   model_name = "facebook/opt-125m"  # Instead of opt-350m
   ```

2. **Reduce batch size** in examples:
   ```python
   batch_size = 16  # Instead of 32
   ```

3. **Close other GPU applications:**
   ```bash
   nvidia-smi  # Check what's using GPU
   ```

4. **Clear GPU cache** before running:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Verification

Run this verification script to check your setup:

```python
import sys

print("Python version:", sys.version)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import bitsandbytes
    print(f"✓ bitsandbytes {bitsandbytes.__version__}")
except ImportError as e:
    print(f"✗ bitsandbytes: {e}")

try:
    import accelerate
    print(f"✓ Accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"✗ Accelerate: {e}")

print("\nSetup verification complete!")
```

Save this as `verify_setup.py` and run:
```bash
python verify_setup.py
```

## Quick Start

Once setup is complete:

```bash
# Run examples in order
python examples/01_basic_int8_quantization.py
python examples/02_4bit_quantization.py
python examples/03_llm_bitsandbytes.py
python examples/04_gptq_quantization.py
python examples/05_comparison_benchmark.py
```

Each example is self-contained and includes detailed explanations.

## GPU Memory Requirements by Example

| Example | Minimum VRAM | Recommended VRAM | Notes |
|---------|--------------|------------------|-------|
| Example 1 | 2GB | 4GB | Basic quantization concepts |
| Example 2 | 2GB | 4GB | 4-bit quantization |
| Example 3 | 4GB | 8GB | Real LLM (opt-350m) |
| Example 4 | 4GB | 8GB | GPTQ concepts |
| Example 5 | 4GB | 8GB | Comprehensive benchmark |

## Common Issues

### ImportError: cannot import name 'XYZ'

Update to latest versions:
```bash
pip install --upgrade transformers accelerate bitsandbytes
```

### Model Download Failures

Set cache directory:
```bash
export HF_HOME=/path/to/cache  # Linux/Mac
set HF_HOME=C:\path\to\cache   # Windows
```

Or use offline mode after first download:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name", local_files_only=True)
```

### Slow Downloads

Use a mirror or download models manually:
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download facebook/opt-350m
```

## Getting Help

If you encounter issues:

1. Check this setup guide
2. Review the example scripts' error messages
3. Check PyTorch/transformers documentation
4. Open an issue on GitHub with:
   - Your hardware specs
   - Python/CUDA/PyTorch versions
   - Full error message
   - Steps to reproduce

## Next Steps

After successful setup:

1. Read the main [README.md](README.md) for tutorial overview
2. Start with Example 1 to learn quantization basics
3. Progress through examples in order
4. Experiment with different models and parameters

Happy quantizing! 🚀
