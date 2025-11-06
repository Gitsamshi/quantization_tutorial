# LLM Quantization Tutorial

A comprehensive, hands-on tutorial for understanding and implementing Large Language Model (LLM) quantization with executable GPU examples.

## Table of Contents

1. [Introduction](#introduction)
2. [What is Quantization?](#what-is-quantization)
3. [Why Quantization Matters](#why-quantization-matters)
4. [Types of Quantization](#types-of-quantization)
5. [Setup Instructions](#setup-instructions)
6. [Tutorial Examples](#tutorial-examples)
7. [Performance Benchmarks](#performance-benchmarks)
8. [References](#references)

## Introduction

This tutorial provides practical, executable examples for quantizing Large Language Models (LLMs) to reduce memory footprint and increase inference speed on GPUs. All examples are designed to run on CUDA-enabled GPUs.

## What is Quantization?

Quantization is the process of reducing the precision of numerical values in a neural network model. In the context of LLMs:

- **Original Models**: Typically use 32-bit floating-point (FP32) or 16-bit floating-point (FP16) numbers
- **Quantized Models**: Use lower-precision formats like 8-bit integers (INT8) or even 4-bit representations

### Mathematical Foundation

The basic quantization process involves mapping floating-point values to discrete integer values:

```
Q(x) = round(x / S) - Z
```

Where:
- `x` is the original floating-point value
- `S` is the scale factor
- `Z` is the zero-point
- `Q(x)` is the quantized integer value

Dequantization reverses this process:

```
x̂ = S * (Q(x) + Z)
```

## Why Quantization Matters

### 1. **Memory Reduction**
- FP32 → INT8: 4× memory reduction
- FP32 → INT4: 8× memory reduction
- Enables running larger models on consumer hardware

### 2. **Speed Improvements**
- Integer operations are faster than floating-point operations
- Better cache utilization due to smaller memory footprint
- Can achieve 2-4× inference speedup

### 3. **Cost Efficiency**
- Lower memory requirements = cheaper GPU deployment
- Reduced cloud computing costs
- Enables edge deployment

### Example: Memory Savings

For a 7B parameter model (7 billion parameters):

| Precision | Bytes/Param | Total Memory | Reduction |
|-----------|-------------|--------------|-----------|
| FP32      | 4 bytes     | 28 GB       | -         |
| FP16      | 2 bytes     | 14 GB       | 2×        |
| INT8      | 1 byte      | 7 GB        | 4×        |
| INT4      | 0.5 bytes   | 3.5 GB      | 8×        |

## Types of Quantization

### 1. **Post-Training Quantization (PTQ)**
- Quantize an already-trained model
- No retraining required
- Quick to implement
- May have some accuracy loss

### 2. **Quantization-Aware Training (QAT)**
- Train model with quantization in mind
- Better accuracy preservation
- More computationally expensive

### 3. **Dynamic Quantization**
- Quantize activations dynamically at runtime
- Weights are quantized statically
- Good balance between speed and accuracy

### 4. **Static Quantization**
- Both weights and activations are quantized
- Requires calibration dataset
- Best performance gains

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (with compute capability 7.0+)
- CUDA Toolkit 11.8+ or 12.0+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantization_tutorial

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM
- RAM: 16GB system RAM

**Recommended:**
- GPU: NVIDIA RTX 3090, A100, or better
- RAM: 32GB+ system RAM

## Tutorial Examples

### Example 1: Basic INT8 Quantization
**File:** `examples/01_basic_int8_quantization.py`

Learn the fundamentals of symmetric and asymmetric INT8 quantization with PyTorch.

```bash
python examples/01_basic_int8_quantization.py
```

### Example 2: 4-bit Quantization
**File:** `examples/02_4bit_quantization.py`

Implement 4-bit quantization for maximum memory savings.

```bash
python examples/02_4bit_quantization.py
```

### Example 3: Real LLM with bitsandbytes
**File:** `examples/03_llm_bitsandbytes.py`

Quantize and run actual LLM models using the popular bitsandbytes library.

```bash
python examples/03_llm_bitsandbytes.py
```

### Example 4: GPTQ Quantization
**File:** `examples/04_gptq_quantization.py`

Advanced quantization using GPTQ (Generative Pre-trained Transformer Quantization).

```bash
python examples/04_gptq_quantization.py
```

### Example 5: Quantization Comparison
**File:** `examples/05_comparison_benchmark.py`

Compare different quantization methods side-by-side with benchmarks.

```bash
python examples/05_comparison_benchmark.py
```

## Performance Benchmarks

Typical results you can expect (tested on NVIDIA RTX 3090):

| Model | Precision | Memory | Speed (tokens/sec) | Perplexity |
|-------|-----------|--------|-------------------|------------|
| LLaMA-7B | FP16 | 14 GB | 25 | 5.68 |
| LLaMA-7B | INT8 | 7 GB | 45 | 5.72 |
| LLaMA-7B | INT4 | 3.5 GB | 60 | 5.89 |

## Key Concepts Covered

1. **Symmetric vs Asymmetric Quantization**
2. **Per-tensor vs Per-channel Quantization**
3. **Calibration Techniques**
4. **Mixed-precision Quantization**
5. **Quantization Error Analysis**
6. **Hardware-specific Optimizations**

## Common Pitfalls and Solutions

### Problem: Accuracy Degradation
**Solution:** Use per-channel quantization and careful calibration

### Problem: Slow Inference Despite Quantization
**Solution:** Ensure you're using optimized kernels (TensorRT, ONNXRuntime)

### Problem: Out of Memory During Quantization
**Solution:** Use gradient checkpointing and batch the quantization process

## Advanced Topics

- **GGML/GGUF Format**: Specialized quantization for llama.cpp
- **AWQ (Activation-aware Weight Quantization)**: Preserve important weights
- **SmoothQuant**: Address outliers in activations
- **ZeroQuant**: Microsoft's zero-shot quantization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## References

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

## License

MIT License - see LICENSE file for details

---

**Happy Quantizing! 🚀**
