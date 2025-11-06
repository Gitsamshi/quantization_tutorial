# Quick Start Guide

Get started with LLM quantization in 5 minutes!

## 1. Prerequisites Check

```bash
# Check if you have NVIDIA GPU with CUDA
nvidia-smi

# Check Python version (need 3.8+)
python --version
```

## 2. Installation

```bash
# Clone and navigate to repo
git clone <repository-url>
cd quantization_tutorial

# Install dependencies (choose one)

# Option A: Full installation (recommended)
pip install -r requirements.txt

# Option B: Minimal installation (for basic examples)
pip install torch numpy

# Option C: Using conda
conda create -n quant python=3.10
conda activate quant
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers accelerate bitsandbytes
```

## 3. Verify Setup

```bash
python verify_setup.py
```

If you see "✓ SETUP COMPLETE", you're ready to go!

## 4. Run Your First Example

```bash
python examples/01_basic_int8_quantization.py
```

This will teach you:
- What quantization is
- Symmetric vs asymmetric quantization
- Per-tensor vs per-channel quantization
- How to quantize neural networks

**Expected runtime:** 1-2 minutes

## 5. Example Progression

### Example 1: Basic INT8 Quantization (Start here!)
```bash
python examples/01_basic_int8_quantization.py
```
**What you'll learn:** Fundamentals of quantization
**Requirements:** Any GPU or CPU
**Runtime:** ~1-2 min

### Example 2: 4-bit Quantization
```bash
python examples/02_4bit_quantization.py
```
**What you'll learn:** Advanced 4-bit techniques, NF4
**Requirements:** Any GPU or CPU
**Runtime:** ~1-2 min

### Example 3: Real LLM with bitsandbytes (Most practical!)
```bash
python examples/03_llm_bitsandbytes.py
```
**What you'll learn:** Quantize actual language models
**Requirements:** 6GB+ GPU, transformers + bitsandbytes
**Runtime:** ~3-5 min (first run downloads model)

### Example 4: GPTQ Quantization
```bash
python examples/04_gptq_quantization.py
```
**What you'll learn:** Production-grade quantization
**Requirements:** GPU recommended
**Runtime:** ~2-3 min

### Example 5: Comprehensive Benchmark
```bash
python examples/05_comparison_benchmark.py
```
**What you'll learn:** Compare all methods side-by-side
**Requirements:** 6GB+ GPU recommended
**Runtime:** ~3-5 min

## Common Commands

```bash
# Check GPU status
nvidia-smi

# Clear GPU memory if needed
python -c "import torch; torch.cuda.empty_cache()"

# Run all examples in sequence
for i in {01..05}; do python examples/${i}_*.py; done

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python examples/03_llm_bitsandbytes.py
```

## What You'll Learn

### Theory (15 minutes)
- ✓ What is quantization and why it matters
- ✓ Different quantization techniques (INT8, INT4, NF4)
- ✓ Trade-offs between accuracy and efficiency
- ✓ When to use which quantization method

### Practice (30 minutes)
- ✓ Implement quantization from scratch
- ✓ Quantize real language models
- ✓ Benchmark different approaches
- ✓ Deploy quantized models

## Troubleshooting

### "CUDA not available"
```bash
# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Out of memory"
Use a smaller model in Example 3:
```python
model_name = "facebook/opt-125m"  # Instead of opt-350m
```

### "bitsandbytes not found"
```bash
# Linux/WSL2
pip install bitsandbytes

# Windows (native) - not fully supported
# Use WSL2 or skip Example 3
```

### "Model download too slow"
First run downloads models (~700MB). Subsequent runs use cache.
Set cache location:
```bash
export HF_HOME=/path/to/fast/storage
```

## Quick Reference

### Memory Requirements

| Model Size | FP32 | FP16 | INT8 | INT4 |
|------------|------|------|------|------|
| 1B params  | 4 GB | 2 GB | 1 GB | 0.5 GB |
| 7B params  | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B params | 52 GB | 26 GB | 13 GB | 6.5 GB |

### Typical Accuracy

| Method | Accuracy | Use Case |
|--------|----------|----------|
| FP16 | 99.9%+ | Maximum accuracy |
| INT8 | 98-99% | General inference |
| INT4/NF4 | 95-98% | Memory-constrained |

### Speed Comparison

| Method | Relative Speed |
|--------|---------------|
| FP32 | 1.0× (baseline) |
| FP16 | 1.5-2× faster |
| INT8 | 2-3× faster |
| INT4 (GPTQ) | 2-4× faster |

## Next Steps

After completing the examples:

1. **Experiment with your own models**
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("your-model-here")
   ```

2. **Try different quantization configs**
   - Different block sizes
   - Mixed precision
   - Custom quantization schemes

3. **Explore advanced topics**
   - QLoRA for fine-tuning
   - GGML/GGUF for llama.cpp
   - AWQ quantization
   - Deployment optimization

4. **Read the papers**
   - LLM.int8: https://arxiv.org/abs/2208.07339
   - GPTQ: https://arxiv.org/abs/2210.17323
   - QLoRA: https://arxiv.org/abs/2305.14314

## Need Help?

- 📖 Full documentation: [README.md](README.md)
- 🔧 Setup issues: [SETUP.md](SETUP.md)
- 💻 Code examples: [examples/](examples/)
- 🐛 Report bugs: GitHub Issues

## Cheat Sheet

```python
# Load model in different precisions

# FP16 (baseline)
model = AutoModelForCausalLM.from_pretrained("model-name", torch_dtype=torch.float16)

# INT8 (bitsandbytes)
model = AutoModelForCausalLM.from_pretrained("model-name", load_in_8bit=True)

# NF4 (bitsandbytes)
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("model-name", quantization_config=config)

# GPTQ (pre-quantized)
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("TheBloke/model-name-GPTQ")
```

---

**Ready to start?** Run Example 1 now:
```bash
python examples/01_basic_int8_quantization.py
```

**Total learning time:** ~1 hour for all examples
**Skill level after completion:** Intermediate quantization practitioner

Good luck! 🚀
