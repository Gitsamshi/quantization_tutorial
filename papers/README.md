# LLM Quantization Research Papers

This collection contains foundational and cutting-edge research papers on Large Language Model (LLM) quantization. These papers represent the evolution of quantization techniques from 2022 to 2025, showcasing different approaches to making LLMs more efficient and deployable on resource-constrained hardware.

## Overview

Quantization is a critical technique for deploying large language models in production environments. By reducing the precision of model weights and activations from 32-bit or 16-bit floating-point to lower bit representations (8-bit, 4-bit, or even 2-bit), we can:

- **Reduce memory footprint** by 2-8x or more
- **Increase inference speed** by 2-4x
- **Enable deployment** on consumer hardware and edge devices
- **Lower operational costs** for serving LLMs at scale

However, quantization introduces accuracy degradation. The papers in this collection represent the state-of-the-art techniques for minimizing this degradation while maximizing efficiency gains.

## Papers by Category

### Foundational Papers & Quantization-Aware Training

#### 11. Quantization and Training of Neural Networks (CVPR 2018)
**File:** `11_Jacob_QAT_CVPR2018.pdf`
**Authors:** Benoit Jacob et al. (Google)
**arXiv:** [1712.05877](https://arxiv.org/abs/1712.05877)

**Key Contribution:** Foundational work introducing quantization-aware training (QAT) with integer-only arithmetic for efficient inference.

**Innovation:**
- Simulated quantization during training
- Integer-only inference on commonly available hardware
- Established best practices for quantizing deep neural networks

**Impact:** Set the foundation for modern quantization techniques. Highly cited classic that introduced key concepts still used today.

---

### Weight-Only Quantization Methods

#### 2. GPTQ (ICLR 2023)
**File:** `02_GPTQ_ICLR2023.pdf`
**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**arXiv:** [2210.17323](https://arxiv.org/abs/2210.17323)

**Key Contribution:** One-shot weight quantization using approximate second-order information. Can compress 175B parameter models in ~4 GPU hours to 3-4 bits.

**Innovation:**
- Layer-wise quantization with Hessian-based optimization
- Extreme compression (3-4 bits) with minimal accuracy loss
- Enables 175B models to fit in a single GPU

**Impact:** Made extreme quantization (sub-4-bit) practically viable for the first time.

---

#### 4. AWQ - Activation-Aware Weight Quantization (MLSys 2024 - Best Paper)
**File:** `04_AWQ_MLSys2024.pdf`
**Authors:** Ji Lin et al. (MIT, NVIDIA)
**arXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)

**Key Contribution:** Protects only 1% of salient weight channels based on activation distributions for superior 4-bit quantization.

**Innovation:**
- Activation-aware channel importance identification
- Hardware-friendly weight-only quantization
- TinyChat framework: 3x+ speedup on GPUs

**Impact:** Best-in-class 4-bit quantization. MLSys 2024 Best Paper Award. Widely adopted in production.

---

#### 8. OmniQuant (ICLR 2024)
**File:** `08_OmniQuant_ICLR2024.pdf`
**Authors:** Wenqi Shao et al.
**arXiv:** [2308.13137](https://arxiv.org/abs/2308.13137)

**Key Contribution:** First differentiable post-training quantization algorithm for LLMs with omnidirectional calibration.

**Innovation:**
- Learnable weight clipping and equivalent transformations
- Superior performance across W4A4, W6A6, W4A16, W3A16, W2A16
- Block-wise error minimization

**Impact:** State-of-the-art PTQ method that works across diverse quantization configurations.

---

#### 10. SpQR - Sparse-Quantized Representation (ICLR 2024)
**File:** `10_SpQR_ICLR2024.pdf`
**Authors:** Tim Dettmers et al.
**arXiv:** [2306.03078](https://arxiv.org/abs/2306.03078)

**Key Contribution:** Near-lossless LLM compression by isolating outlier weights in higher precision and compressing others to 3-4 bits.

**Innovation:**
- Identifies and isolates outlier weights
- 3.4x+ memory reduction without degradation
- 20-30% faster inference than FP16

**Impact:** Achieves near-lossless compression with practical speedups.

---

### Weight + Activation Quantization

#### 1. LLM.int8() (NeurIPS 2022)
**File:** `01_LLM_int8_NeurIPS2022.pdf`
**Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
**arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339)

**Key Contribution:** 8-bit matrix multiplication for transformers using mixed-precision decomposition to handle outlier features.

**Innovation:**
- Vector-wise quantization with separate normalization
- Mixed-precision: 16-bit for outliers, 8-bit for everything else
- First to quantize models beyond 6.7B parameters without loss

**Impact:** Enabled deployment of 175B models on consumer hardware.

---

#### 3. SmoothQuant (ICML 2023)
**File:** `03_SmoothQuant_ICML2023.pdf`
**Authors:** Guangxuan Xiao et al. (MIT, NVIDIA)
**arXiv:** [2211.10438](https://arxiv.org/abs/2211.10438)

**Key Contribution:** Training-free W8A8 quantization by smoothing activation outliers offline.

**Innovation:**
- Migrates quantization difficulty from activations to weights
- Per-channel scaling factors
- 1.56x speedup and 2x memory reduction

**Impact:** First successful W8A8 quantization across all LLM operations for hardware accelerators.

---

#### 12. Understanding INT4 Quantization for Transformers (2023)
**File:** `12_INT4_Quantization_2023.pdf`
**Authors:** Guangxuan Xiao et al.
**arXiv:** [2301.12017](https://arxiv.org/abs/2301.12017)

**Key Contribution:** Comprehensive study of INT4 W4A4 quantization for encoder, encoder-decoder, and decoder models.

**Innovation:**
- 8.5x faster latency for INT4 vs FP16
- No loss for BERT, negligible for BART, manageable for GPT
- Detailed failure case analysis

**Impact:** Demonstrated viability of INT4 inference with significant speedups.

---

#### 15. ZeroQuant (2022)
**File:** `15_ZeroQuant_2022.pdf`
**Authors:** Yuxiong He et al. (Microsoft)
**arXiv:** [2206.01861](https://arxiv.org/abs/2206.01861)

**Key Contribution:** Efficient and affordable post-training quantization with group-wise quantization and layer-wise knowledge distillation.

**Innovation:**
- Group-wise quantization for activations
- INT4/INT8 mixed-precision with knowledge distillation
- Up to 5.19x speedup on A100 GPUs

**Impact:** Production-ready PTQ system from Microsoft for efficient LLM deployment.

---

#### 16. I-BERT - Integer-Only BERT (ICML 2021)
**File:** `16_I-BERT_ICML2021.pdf`
**Authors:** Sehoon Kim et al. (UC Berkeley)
**arXiv:** [2101.01321](https://arxiv.org/abs/2101.01321)

**Key Contribution:** Integer-only arithmetic for entire BERT inference including nonlinear operations.

**Innovation:**
- Integer-only approximations for GELU, Softmax, LayerNorm
- 2.4-4.0x speedup for INT8 on T4 GPUs
- Similar or better accuracy than FP baseline

**Impact:** Demonstrated that even complex nonlinear operations can be efficiently quantized.

---

#### 6. SpinQuant (ICLR 2025)
**File:** `06_SpinQuant_ICLR2025.pdf`
**Authors:** Zechun Liu et al. (Meta AI)
**arXiv:** [2405.16406](https://arxiv.org/abs/2405.16406)

**Key Contribution:** Learned rotation matrices for 4-bit quantization of weights, activations, and KV-cache.

**Innovation:**
- Rotation matrices preserving full-precision outputs
- Comprehensive 4-bit quantization (weights + activations + KV)
- Only 2.9 point gap vs. full precision on LLaMA-2 7B

**Impact:** State-of-the-art comprehensive 4-bit quantization as of 2025.

---

### KV-Cache Quantization

#### 13. KVQuant (NeurIPS 2024)
**File:** `13_KVQuant_NeurIPS2024.pdf`
**Authors:** Coleman Hooper et al. (SqueezeAI Lab)

**Key Contribution:** Enables 10 million context length LLM inference through KV cache quantization.

**Innovation:**
- Per-Channel Key Quantization and Pre-RoPE Key Quantization
- <0.1 perplexity degradation with 3-bit quantization
- Up to 10M context on 8-GPU system
- Custom CUDA kernels with 1.7x speedup

**Impact:** Breakthrough for long-context LLM inference with extreme memory efficiency.

---

#### 21. KV Cache is 1 Bit Per Channel (NeurIPS 2024)
**File:** `21_1bit_KV_Cache_NeurIPS2024.pdf`
**Authors:** Tianyi Zhang et al.

**Key Contribution:** Coupled Quantization (CQ) exploiting channel interdependence for extreme KV cache compression.

**Innovation:**
- Couples multiple channels for joint quantization
- 1.4-3.5x throughput improvement
- Preserves quality down to 1-bit per channel

**Impact:** Pushes KV cache quantization to extreme limits while maintaining model quality.

---

### Extreme Quantization (≤2-bit)

#### 5. QuIP - 2-Bit Quantization with Guarantees (2023)
**File:** `05_QuIP_2023.pdf`
**Authors:** Jerry Chee et al. (Cornell)
**arXiv:** [2307.13304](https://arxiv.org/abs/2307.13304)

**Key Contribution:** First viable 2-bit LLM quantization using lattice codebooks and incoherence processing.

**Innovation:**
- E8 lattice codebooks for vector quantization
- Incoherence processing via randomized Hadamard transforms
- Theoretical guarantees on reconstruction error

**Impact:** Pushed quantization boundary to 2 bits (16x compression) with usable performance.

---

#### 19. QuIP# - Even Better LLM Quantization (2024)
**File:** `19_QuIP_Sharp_2024.pdf`
**Authors:** Albert Tseng et al. (Cornell)
**arXiv:** [2402.04396](https://arxiv.org/abs/2402.04396)

**Key Contribution:** Improved QuIP with better incoherence processing and E8 lattice codebooks.

**Innovation:**
- Randomized Hadamard transform (faster, better theory)
- Hardware-efficient E8 lattice codebooks
- First PTQ where 3-bit outscales 4-bit

**Impact:** State-of-the-art for extreme (<4-bit) quantization.

---

#### 14. BiLLM - 1-Bit Post-Training Quantization (ICML 2024)
**File:** `14_BiLLM_ICML2024.pdf`
**Authors:** Wei Huang et al.
**arXiv:** [2402.04291](https://arxiv.org/abs/2402.04291)

**Key Contribution:** First 1-bit PTQ scheme for LLMs achieving high-accuracy inference with only 1.08-bit average.

**Innovation:**
- Structural salient weight selection
- Binary residual approximation strategy
- Optimal splitting search for non-salient weights
- Achieves 8.41 perplexity on LLaMA2-70B

**Impact:** Groundbreaking 1-bit quantization enabling extreme compression scenarios.

---

#### 20. AQLM - Extreme Compression via Additive Quantization (ICML 2024)
**File:** `20_AQLM_ICML2024.pdf`
**Authors:** Vage Egiazarian et al.
**arXiv:** [2401.06118](https://arxiv.org/abs/2401.06118)

**Key Contribution:** Multi-codebook quantization for extreme LLM compression (2-3 bits).

**Innovation:**
- Learned additive quantization in input-adaptive fashion
- Joint optimization across transformer blocks
- Pareto optimal for <3 bits per parameter

**Impact:** Best quality in extreme compression (2-bit) regime.

---

### Fine-Tuning with Quantization

#### 9. QLoRA - Efficient Finetuning of Quantized LLMs (2023)
**File:** `09_QLoRA_2023.pdf`
**Authors:** Tim Dettmers, Artidoro Pagnoni
**arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)

**Key Contribution:** Enables finetuning 65B models on a single 48GB GPU through 4-bit quantization + LoRA.

**Innovation:**
- 4-bit NormalFloat (NF4) data type
- Double quantization to reduce memory
- Paged optimizers
- Backpropagates through frozen 4-bit model into LoRA adapters

**Impact:** Democratized LLM finetuning. Widely adopted for efficient model adaptation.

---

#### 18. LoftQ - LoRA-Fine-Tuning-Aware Quantization (ICLR 2024)
**File:** `18_LoftQ_ICLR2024.pdf`
**Authors:** Yixiao Li et al. (Microsoft)
**arXiv:** [2310.08659](https://arxiv.org/abs/2310.08659)

**Key Contribution:** Simultaneously quantizes LLM and finds optimal low-rank initialization for LoRA.

**Innovation:**
- Quantization-aware LoRA initialization
- Reduces discrepancy between quantized and full-precision
- Superior in 2-bit and 2/4-bit mixed precision

**Impact:** Improved QLoRA approach with better initialization strategy.

---

### Mixed-Precision & Hardware-Aware Quantization

#### 17. HAWQ - Hessian Aware Quantization (ICCV 2019)
**File:** `17_HAWQ_ICCV2019.pdf`
**Authors:** Zhen Dong et al. (UC Berkeley)
**PDF:** [Berkeley](https://www.stat.berkeley.edu/~mmahoney/pubs/HAWQ_ICCV_2019_paper.pdf)

**Key Contribution:** Mixed-precision quantization using Hessian spectrum to determine per-layer precision.

**Innovation:**
- Second-order sensitivity analysis via Hessian
- Automatic per-layer bit-width selection
- Hardware-aware mixed-precision optimization

**Impact:** Pioneered principled mixed-precision quantization based on theoretical foundations.

---

### Survey & Comprehensive Studies

#### 7. A Comprehensive Study on Quantization Techniques for LLMs (2024)
**File:** `07_Comprehensive_Study_Quantization_2024.pdf`
**Authors:** Jiedong Lang, Zhehao Guo, Shuyu Huang
**arXiv:** [2411.02530](https://arxiv.org/abs/2411.02530)

**Key Contribution:** Comprehensive survey covering mathematical foundations, taxonomy, and practical considerations.

**Coverage:**
- Mathematical foundations of quantization
- PTQ vs. QAT comparison
- KV-cache quantization
- Performance analysis across methods

**Impact:** Essential reference for understanding the quantization landscape.

---

## Evolution and Trends

### Timeline of Innovation

**2018-2019:** Foundations
- **Jacob et al. (CVPR 2018)**: QAT foundations and integer-only inference
- **HAWQ (ICCV 2019)**: Hessian-based mixed-precision quantization

**2021-2022:** Early LLM quantization
- **I-BERT (ICML 2021)**: Integer-only BERT with nonlinear operation approximations
- **ZeroQuant (2022)**: Group-wise quantization with knowledge distillation
- **LLM.int8() (NeurIPS 2022)**: Mixed-precision 8-bit with outlier handling
- **GPTQ (ICLR 2023)**: Extreme 3-4 bit weight-only quantization

**2023:** Breakthrough year
- **SmoothQuant (ICML 2023)**: W8A8 quantization smoothing activation outliers
- **INT4 Quantization (2023)**: Comprehensive INT4 study across architectures
- **QuIP (2023)**: Lattice-based 2-bit quantization
- **QLoRA (2023)**: 4-bit finetuning democratization

**2024:** Specialization and extreme compression
- **AWQ (MLSys 2024)**: Activation-aware weight quantization (Best Paper)
- **OmniQuant (ICLR 2024)**: Differentiable PTQ across configurations
- **SpQR (ICLR 2024)**: Near-lossless compression with outlier isolation
- **LoftQ (ICLR 2024)**: Quantization-aware LoRA initialization
- **QuIP# (2024)**: Improved 2-bit quantization
- **BiLLM (ICML 2024)**: Groundbreaking 1-bit quantization
- **AQLM (ICML 2024)**: Additive quantization for 2-3 bit compression
- **KVQuant (NeurIPS 2024)**: 10M context through KV cache quantization
- **1-bit KV Cache (NeurIPS 2024)**: Extreme KV cache compression

**2025:** State-of-the-art
- **SpinQuant (ICLR 2025)**: Learned rotations for comprehensive 4-bit quantization

### Key Research Directions

1. **Weight-only vs. Weight+Activation Quantization**
   - Weight-only (GPTQ, AWQ, OmniQuant, SpQR): Easier to implement, excellent for memory reduction
   - W8A8 (SmoothQuant, ZeroQuant): Better hardware utilization, faster inference
   - W4A4 (INT4, SpinQuant): Extreme efficiency with careful design

2. **Outlier Handling Strategies**
   - Mixed-precision (LLM.int8()): Keep outliers in FP16
   - Smoothing/migration (SmoothQuant): Shift difficulty from activations to weights
   - Channel protection (AWQ): Protect salient 1% of channels
   - Sparse representation (SpQR): Isolate outliers, compress rest
   - Rotations (SpinQuant): Learned transformations for better quantizability

3. **Extreme Quantization (≤2 bits)**
   - **GPTQ**: 3-4 bits (widely adopted)
   - **QuIP/QuIP#**: 2 bits with lattice codebooks
   - **AQLM**: 2-3 bits via additive quantization (Pareto optimal)
   - **BiLLM**: 1 bit with binary residuals (cutting edge)
   - Trade-offs between compression ratio and model quality

4. **KV-Cache Quantization**
   - Critical for long-context inference
   - **KVQuant**: 3-bit with 10M context capability
   - **1-bit KV Cache**: Extreme compression via coupled quantization
   - Different characteristics than weight quantization
   - Enables much longer context windows

5. **Fine-tuning with Quantization**
   - **QLoRA**: 4-bit quantization + LoRA (widely adopted)
   - **LoftQ**: Better initialization for quantized LoRA
   - Enables training on consumer hardware

6. **Theoretical Foundations**
   - **HAWQ**: Hessian-based sensitivity analysis
   - **QuIP/QuIP#**: Rate-distortion theory and lattice quantization
   - **AQLM**: Multi-codebook optimization theory
   - Moving from heuristics to principled approaches

## Practical Recommendations

### For Deployment

**Choose based on your constraints:**

- **Production LLMs (7B-70B), balanced approach**:
  - AWQ or GPTQ at 4-bit for best accuracy/efficiency trade-off
  - OmniQuant for flexible configurations (W4A4, W6A6, etc.)

- **Hardware accelerators with INT8 support**:
  - SmoothQuant W8A8 for maximum throughput
  - ZeroQuant for Microsoft ecosystem

- **Memory-constrained edge devices**:
  - GPTQ 3-bit for reasonable quality
  - QuIP# 2-bit for extreme memory limits
  - SpQR for near-lossless with 3-4 bits + sparse outliers

- **Long-context applications**:
  - KVQuant for contexts up to 10M tokens
  - 1-bit KV Cache for extreme KV compression

- **Fine-tuning on limited hardware**:
  - QLoRA (widely supported, mature)
  - LoftQ (better initialization for 2-bit scenarios)

- **Cutting-edge research models**:
  - SpinQuant 4-bit for state-of-the-art 2025 results

**Extreme compression scenarios**:
- AQLM for 2-3 bit (best Pareto frontier)
- BiLLM for 1-bit (experimental, extreme cases)

### For Research

**Learning path:**

1. **Start with foundations**:
   - Jacob et al. (2018) for QAT fundamentals
   - Comprehensive Survey (Paper #7) for landscape overview

2. **Core techniques**:
   - LLM.int8() for mixed-precision concepts
   - GPTQ for weight-only quantization
   - SmoothQuant for activation quantization

3. **Advanced methods**:
   - HAWQ for theoretical foundations (Hessian-based)
   - AWQ for activation-aware techniques
   - SpQR for outlier handling

4. **Extreme quantization**:
   - QuIP/QuIP# for 2-bit with theory
   - AQLM for multi-codebook approaches
   - BiLLM for 1-bit frontier

5. **Specialized areas**:
   - KVQuant and 1-bit KV Cache for long-context
   - QLoRA and LoftQ for parameter-efficient finetuning
   - OmniQuant for differentiable PTQ

6. **State-of-the-art**:
   - SpinQuant for latest 2025 benchmarks

## Tools and Frameworks

Many of these papers come with open-source implementations:

- **LLM.int8()**: Integrated into Hugging Face `transformers` and `bitsandbytes`
- **GPTQ**: `auto-gptq`, `exllama`, `llama.cpp`
- **AWQ**: `llm-awq`, `vLLM`, `TinyChat`
- **SmoothQuant**: `smoothquant` library, integrated into PyTorch
- **QLoRA**: Part of `bitsandbytes` and `peft` libraries
- **SpQR**: Official GitHub implementation
- **OmniQuant**: Official GitHub repository
- **KVQuant**: Official GitHub from SqueezeAI Lab
- **QuIP/QuIP#**: Cornell-RelaxML repositories
- **AQLM**: Official PyTorch implementation
- **BiLLM**: Official GitHub repository
- **SpinQuant**: Official implementation on GitHub
- **LoftQ**: Official Microsoft Research implementation

## Future Directions

Based on these papers, emerging trends include:

1. **Learned transformations** (SpinQuant) beyond hand-crafted heuristics
2. **Multi-modal quantization** for vision-language models (AWQ pioneering this)
3. **Dynamic quantization** adapting to input characteristics
4. **Hardware-software co-design** for optimal efficiency
5. **Sub-2-bit quantization** (BiLLM at 1-bit shows this is possible)
6. **Long-context optimization** (KVQuant enabling 10M+ contexts)
7. **Hybrid approaches** combining multiple techniques (e.g., SpQR's sparse + dense)
8. **Theoretically-principled methods** replacing heuristics (QuIP#, AQLM, HAWQ)
9. **Training-aware quantization** for even better quality (beyond just inference)
10. **Specialized quantization** for different model components

## Quick Reference Table

| Paper | Year | Venue | Bits | Type | Key Innovation | Best For |
|-------|------|-------|------|------|----------------|----------|
| Jacob QAT | 2018 | CVPR | 8 | W+A | Integer-only inference | QAT foundations |
| HAWQ | 2019 | ICCV | Mixed | W+A | Hessian-based precision | Mixed-precision theory |
| I-BERT | 2021 | ICML | 8 | W+A | Integer nonlinear ops | BERT quantization |
| ZeroQuant | 2022 | - | 4/8 | W+A | Group-wise + KD | Microsoft ecosystem |
| LLM.int8() | 2022 | NeurIPS | 8 | W+A | Mixed-precision outliers | Consumer hardware |
| GPTQ | 2023 | ICLR | 3-4 | W | Hessian-based PTQ | Extreme weight compression |
| SmoothQuant | 2023 | ICML | 8 | W+A | Activation smoothing | Hardware accelerators |
| INT4 Study | 2023 | - | 4 | W+A | Comprehensive analysis | Understanding INT4 |
| QuIP | 2023 | - | 2 | W | Lattice codebooks | 2-bit compression |
| QLoRA | 2023 | - | 4 | W | NF4 + LoRA | Finetuning on GPU |
| AWQ | 2024 | MLSys | 4 | W | Activation-aware | Production 4-bit (Best) |
| OmniQuant | 2024 | ICLR | 2-6 | W/W+A | Differentiable PTQ | Flexible configs |
| SpQR | 2024 | ICLR | 3-4 | W | Sparse outliers | Near-lossless |
| LoftQ | 2024 | ICLR | 2-4 | W | QAT LoRA init | Better LoRA |
| QuIP# | 2024 | - | 2-3 | W | Improved QuIP | Extreme compression |
| BiLLM | 2024 | ICML | 1 | W | Binary residuals | 1-bit frontier |
| AQLM | 2024 | ICML | 2-3 | W | Additive quantization | 2-bit Pareto optimal |
| KVQuant | 2024 | NeurIPS | 3 | KV | Per-channel KV | Long context (10M) |
| 1-bit KV | 2024 | NeurIPS | 1 | KV | Coupled quantization | Extreme KV compression |
| SpinQuant | 2025 | ICLR | 4 | W+A+KV | Learned rotations | SOTA 4-bit (2025) |

**Legend:**
- **Type**: W (Weight-only), A (Activation), KV (KV-cache)
- **Bits**: Quantization bit-width

## Citation

If you use these papers in your research, please cite the original authors. BibTeX entries can be found on the respective arXiv pages.

## Additional Resources

- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [Awesome LLM Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [Awesome Quantization Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)

---

## Summary Statistics

- **Total Papers**: 21
- **Date Range**: 2018-2025 (7 years of quantization research)
- **Top Venues**: NeurIPS (3), ICLR (5), ICML (4), CVPR (1), ICCV (1), MLSys (1)
- **Bit Ranges Covered**: 1-bit to 8-bit quantization
- **Categories**:
  - Weight-only quantization: 5 papers
  - Weight + Activation: 7 papers
  - KV-Cache: 2 papers
  - Extreme (≤2-bit): 4 papers
  - Fine-tuning: 2 papers
  - Foundations/Survey: 3 papers

---

**Last Updated:** November 19, 2025
**Collection Size:** 21 high-impact papers (67.8 MB total)
**Maintained by:** This repository contains papers from 2018-2025 representing the complete evolution of neural network and LLM quantization research.
