# LLM Quantization Research Papers

This collection contains foundational and cutting-edge research papers on Large Language Model (LLM) quantization. These papers represent the evolution of quantization techniques from 2022 to 2025, showcasing different approaches to making LLMs more efficient and deployable on resource-constrained hardware.

## Overview

Quantization is a critical technique for deploying large language models in production environments. By reducing the precision of model weights and activations from 32-bit or 16-bit floating-point to lower bit representations (8-bit, 4-bit, or even 2-bit), we can:

- **Reduce memory footprint** by 2-8x or more
- **Increase inference speed** by 2-4x
- **Enable deployment** on consumer hardware and edge devices
- **Lower operational costs** for serving LLMs at scale

However, quantization introduces accuracy degradation. The papers in this collection represent the state-of-the-art techniques for minimizing this degradation while maximizing efficiency gains.

## Papers Included

### 1. LLM.int8() (NeurIPS 2022)
**File:** `01_LLM_int8_NeurIPS2022.pdf`
**Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
**arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339)

**Key Contribution:**
Introduces 8-bit matrix multiplication for transformers that handles emergent outlier features through a mixed-precision decomposition strategy. This method enables running models like OPT-175B and BLOOM-176B with no performance degradation while using half the GPU memory.

**Innovation:**
- Vector-wise quantization with separate normalization constants
- Mixed-precision approach: 16-bit for outlier dimensions, 8-bit for everything else
- First practical method to quantize models beyond 6.7B parameters without accuracy loss

**Impact:** Enabled deployment of massive models on consumer hardware for the first time.

---

### 2. GPTQ (ICLR 2023)
**File:** `02_GPTQ_ICLR2023.pdf`
**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**arXiv:** [2210.17323](https://arxiv.org/abs/2210.17323)

**Key Contribution:**
A one-shot weight quantization method based on approximate second-order information that can compress GPT-style models with 175B parameters in approximately 4 GPU hours to 3-4 bits per weight.

**Innovation:**
- Leverages layer-wise quantization with Hessian-based optimization
- Achieves extreme compression (3-4 bits) with minimal accuracy loss
- Enables 175B parameter models to fit in a single GPU for inference
- 3.25x to 4.5x speedup depending on hardware

**Impact:** Made extreme quantization (sub-4-bit) practically viable for the first time, democratizing access to very large models.

---

### 3. SmoothQuant (ICML 2023)
**File:** `03_SmoothQuant_ICML2023.pdf`
**Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
**arXiv:** [2211.10438](https://arxiv.org/abs/2211.10438)

**Key Contribution:**
A training-free, post-training quantization method that enables W8A8 (8-bit weights and 8-bit activations) for all matrix multiplications in LLMs by smoothing activation outliers offline.

**Innovation:**
- Migrates quantization difficulty from activations to weights through mathematical transformation
- Per-channel scaling factors to handle outliers
- Works across diverse architectures: OPT, BLOOM, GLM, Llama, Falcon, Mistral, Mixtral
- 1.56x speedup and 2x memory reduction with negligible accuracy loss

**Impact:** First method to successfully quantize both weights AND activations to 8-bit across all LLM operations, enabling efficient INT8 inference on hardware accelerators.

---

### 4. AWQ (MLSys 2024 - Best Paper Award)
**File:** `04_AWQ_MLSys2024.pdf`
**Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
**arXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)

**Key Contribution:**
Activation-aware weight quantization that protects only 1% of salient weight channels to achieve superior 4-bit quantization results without backpropagation or reconstruction.

**Innovation:**
- Insight: Not all weights are equally important; identifying and protecting salient channels via activation distributions
- Hardware-friendly weight-only quantization
- Generalizes across domains and modalities (including vision-language models)
- TinyChat framework: 3x+ speedup on mobile and desktop GPUs

**Impact:** Achieved best-in-class 4-bit quantization results with practical inference speedups. Won MLSys 2024 Best Paper Award. Widely adopted in production systems.

---

### 5. QuIP (2023)
**File:** `05_QuIP_2023.pdf`
**Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa
**arXiv:** [2307.13304](https://arxiv.org/abs/2307.13304)

**Key Contribution:**
First viable 2-bit LLM quantization method using lattice codebooks and incoherence processing with theoretical guarantees.

**Innovation:**
- Uses E8 lattice codebooks for vector quantization (better rate-distortion properties)
- Incoherence processing via randomized Hadamard transformations
- Adaptive rounding based on second-order information
- Theoretical guarantees on reconstruction error

**Impact:** Pushed the boundary of extreme quantization to 2 bits per weight, enabling 16x compression ratios while maintaining usable model performance.

---

### 6. SpinQuant (ICLR 2025)
**File:** `06_SpinQuant_ICLR2025.pdf`
**Authors:** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort
**arXiv:** [2405.16406](https://arxiv.org/abs/2405.16406)

**Key Contribution:**
Introduces learned rotation matrices that improve quantization accuracy by up to 13 percentage points on downstream tasks, achieving near-full-precision results with 4-bit quantization of weights, activations, and KV-cache.

**Innovation:**
- Rotation matrices that preserve full-precision outputs while enhancing quantization
- Applies to weights, activations, AND KV-cache simultaneously
- Only 2.9 point accuracy gap vs. full precision on LLaMA-2 7B (zero-shot reasoning)
- State-of-the-art results on latest models

**Impact:** Current state-of-the-art for comprehensive 4-bit quantization. Represents the cutting edge as of 2025.

---

### 7. A Comprehensive Study on Quantization Techniques for LLMs (2024)
**File:** `07_Comprehensive_Study_Quantization_2024.pdf`
**Authors:** Jiedong Lang, Zhehao Guo, Shuyu Huang
**arXiv:** [2411.02530](https://arxiv.org/abs/2411.02530)

**Key Contribution:**
A comprehensive survey and analysis of quantization techniques for LLMs, covering mathematical foundations, implementation strategies, and performance metrics.

**Coverage:**
- Mathematical foundations of quantization
- Taxonomy of quantization approaches
- Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)
- KV-cache quantization techniques
- Performance analysis across different methods
- Practical deployment considerations

**Impact:** Essential reference for understanding the landscape of LLM quantization research and choosing appropriate techniques for specific use cases.

---

## Evolution and Trends

### Timeline of Innovation

**2022:** Foundation era
- **LLM.int8()**: Mixed-precision 8-bit quantization with outlier handling
- **GPTQ**: Extreme 3-4 bit weight-only quantization
- **SmoothQuant**: W8A8 quantization for activations and weights

**2023-2024:** Refinement and specialization
- **AWQ**: Activation-aware channel protection
- **QuIP**: Lattice-based 2-bit quantization
- Hardware-optimized implementations
- KV-cache specific methods

**2024-2025:** Holistic approaches
- **SpinQuant**: Learned rotations for comprehensive 4-bit quantization
- Multi-modal model support
- Production-grade tooling and frameworks

### Key Research Directions

1. **Weight-only vs. Weight+Activation Quantization**
   - Weight-only (GPTQ, AWQ): Easier to implement, good for memory reduction
   - W8A8 (SmoothQuant): Better hardware utilization, faster inference

2. **Outlier Handling Strategies**
   - Mixed-precision (LLM.int8())
   - Smoothing/migration (SmoothQuant)
   - Channel protection (AWQ)
   - Rotations (SpinQuant)

3. **Extreme Quantization (<4 bits)**
   - GPTQ: 3-4 bits
   - QuIP: 2 bits
   - Trade-offs between compression and usability

4. **KV-Cache Quantization**
   - Critical for long-context inference
   - Different characteristics than weight quantization
   - Active area of research

## Practical Recommendations

### For Deployment

- **Production LLMs (7B-70B)**: Start with AWQ or GPTQ at 4-bit for best accuracy/efficiency balance
- **Very large models (>100B)**: Consider SmoothQuant W8A8 for hardware acceleration
- **Memory-constrained edge devices**: GPTQ 3-bit or QuIP 2-bit
- **Latest models with best results**: SpinQuant 4-bit

### For Research

- **Baseline understanding**: Start with the comprehensive survey (Paper #7)
- **Foundational techniques**: Study LLM.int8() and GPTQ first
- **State-of-the-art comparison**: SpinQuant for 2024-2025 benchmarks
- **Novel approaches**: QuIP for lattice-based methods

## Tools and Frameworks

Many of these papers come with open-source implementations:

- **LLM.int8()**: Integrated into Hugging Face `transformers` and `bitsandbytes`
- **GPTQ**: `auto-gptq`, `exllama`, `llama.cpp`
- **AWQ**: `llm-awq`, `vLLM`, `TinyChat`
- **SmoothQuant**: `smoothquant` library, integrated into PyTorch
- **SpinQuant**: Official implementation on GitHub

## Future Directions

Based on these papers, emerging trends include:

1. **Learned transformations** (SpinQuant) beyond hand-crafted heuristics
2. **Multi-modal quantization** for vision-language models
3. **Dynamic quantization** adapting to input characteristics
4. **Hardware-software co-design** for optimal efficiency
5. **Sub-2-bit quantization** with maintained usability
6. **Long-context optimization** through KV-cache compression

## Citation

If you use these papers in your research, please cite the original authors. BibTeX entries can be found on the respective arXiv pages.

## Additional Resources

- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [Awesome LLM Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [Awesome Quantization Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)

---

**Last Updated:** November 2024
**Maintained by:** This repository contains papers from 2022-2025 representing the evolution of LLM quantization research.
