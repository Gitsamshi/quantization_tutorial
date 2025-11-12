# LLM Reinforcement Learning Tutorial

A comprehensive, hands-on tutorial for understanding and implementing Reinforcement Learning from Human Feedback (RLHF) for Large Language Models, covering PPO, DPO, GRPO, and other state-of-the-art algorithms with mathematical derivations and executable examples.

## Table of Contents

1. [Introduction](#introduction)
2. [What is RLHF?](#what-is-rlhf)
3. [Why RLHF Matters for LLMs](#why-rlhf-matters-for-llms)
4. [Core Concepts](#core-concepts)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Algorithms Overview](#algorithms-overview)
7. [Setup Instructions](#setup-instructions)
8. [Tutorial Examples](#tutorial-examples)
9. [References](#references)

## Introduction

This tutorial provides practical, executable examples for training Large Language Models using Reinforcement Learning from Human Feedback (RLHF). We cover state-of-the-art algorithms including PPO, DPO, and GRPO with complete mathematical derivations and implementation details.

## What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is a technique to align language models with human preferences by:

1. **Supervised Fine-Tuning (SFT)**: Train a base model on high-quality demonstrations
2. **Reward Modeling**: Train a reward model to predict human preferences
3. **RL Optimization**: Use RL algorithms to optimize the policy against the reward model

This approach has been crucial for models like ChatGPT, Claude, and GPT-4.

## Why RLHF Matters for LLMs

### Key Benefits

1. **Alignment**: Models learn to generate outputs that humans prefer
2. **Safety**: Reduces harmful or biased outputs
3. **Instruction Following**: Better at following complex instructions
4. **Helpful & Harmless**: Balance between being helpful and avoiding harm

### Example Impact

| Model Type | Helpfulness Score | Safety Score | Instruction Following |
|------------|------------------|--------------|----------------------|
| Base LLM | 6.2/10 | 5.5/10 | 60% |
| SFT Only | 7.5/10 | 7.0/10 | 75% |
| SFT + RLHF | 8.9/10 | 9.2/10 | 92% |

## Core Concepts

### 1. Policy (π)
The language model that generates text. We denote:
- **πθ**: Policy with parameters θ (the model we're training)
- **πref**: Reference policy (usually the SFT model, kept frozen)

### 2. Reward Model (r)
A model trained to predict human preferences:
- **r(x, y)**: Reward for response y given prompt x
- Typically trained on pairwise preference data

### 3. Value Function (V)
Estimates expected future rewards:
- **V(s)**: Expected return from state s

### 4. KL Divergence Constraint
Prevents the policy from deviating too far from the reference:
- **DKL(πθ || πref)**: Measures distribution difference

## Mathematical Foundations

### Standard RL Objective

The goal in RL is to maximize expected reward:

```
J(θ) = E[∑(t=0 to T) γᵗ r(sₜ, aₜ)]
```

Where:
- **θ**: Policy parameters
- **γ**: Discount factor (typically 1.0 for LLMs)
- **r(sₜ, aₜ)**: Reward at time t
- **sₜ**: State (prompt + generated text so far)
- **aₜ**: Action (next token)

### RLHF Objective with KL Penalty

For LLMs, we modify the objective to include a KL penalty:

```
J(θ) = E(x,y)~πθ [r(x, y) - β · DKL(πθ(y|x) || πref(y|x))]
```

Where:
- **x**: Input prompt
- **y**: Generated response
- **β**: KL penalty coefficient (controls exploration vs exploitation)
- **DKL**: KL divergence between current and reference policy

### KL Divergence for Discrete Distributions

For language models (discrete token distributions):

```
DKL(πθ || πref) = ∑ᵢ πθ(yᵢ|x) log(πθ(yᵢ|x) / πref(yᵢ|x))
```

### Policy Gradient Theorem

The gradient of the expected reward is:

```
∇θ J(θ) = E[∑t ∇θ log πθ(aₜ|sₜ) · Aᵗ]
```

Where **Aᵗ** is the advantage function:

```
Aᵗ = Qᵗ - V(sₜ) = r(sₜ, aₜ) + γV(sₜ₊₁) - V(sₜ)
```

## Algorithms Overview

### 1. PPO (Proximal Policy Optimization)

**Paper**: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

PPO is the most widely used algorithm for RLHF. It constrains policy updates using a clipped objective.

#### Mathematical Formulation

The PPO objective is:

```
L^CLIP(θ) = E[min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)]
```

Where:
- **rₜ(θ) = πθ(aₜ|sₜ) / πold(aₜ|sₜ)**: Probability ratio
- **ε**: Clipping parameter (typically 0.2)
- **Âₜ**: Advantage estimate

The full PPO loss for LLMs includes:

```
L(θ) = E[L^CLIP(θ) - β·DKL(πθ || πref) + c₁·L^VF(θ)]
```

Where:
- **L^VF(θ)**: Value function loss (MSE)
- **c₁**: Value function coefficient

#### Advantage Estimation (GAE)

Generalized Advantage Estimation with parameter λ:

```
Âₜ = ∑(l=0 to ∞) (γλ)ˡ δₜ₊ₗ
```

Where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ) is the TD error.

#### Why PPO Works

1. **Clipping prevents large updates**: Ensures training stability
2. **First-order method**: Computationally efficient
3. **Sample efficient**: Reuses data multiple epochs
4. **Proven track record**: Used in ChatGPT, Claude, etc.

### 2. DPO (Direct Preference Optimization)

**Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

DPO eliminates the need for explicit reward modeling and RL by directly optimizing preferences.

#### Key Insight

Under the Bradley-Terry preference model, the optimal policy has a closed form:

```
π*(y|x) = (1/Z(x)) · πref(y|x) · exp(r(x,y)/β)
```

Where Z(x) is the partition function.

#### Reparameterization

Solving for the reward:

```
r(x, y) = β log(π*(y|x)/πref(y|x)) + β log Z(x)
```

#### DPO Objective

Given preference data (x, yₘ, yₗ) where yₘ ≻ yₗ (yₘ preferred over yₗ):

```
L_DPO(θ) = -E[log σ(β log(πθ(yₘ|x)/πref(yₘ|x)) - β log(πθ(yₗ|x)/πref(yₗ|x)))]
```

Where σ is the sigmoid function.

Simplified form:

```
L_DPO(θ) = -E[log σ(β[log πθ(yₘ|x) - log πθ(yₗ|x) - log πref(yₘ|x) + log πref(yₗ|x)])]
```

#### Gradient

The gradient is:

```
∇θ L_DPO = -β E[(σ(Δ̂) - 1)[∇θ log πθ(yₘ|x) - ∇θ log πθ(yₗ|x)]]
```

Where:
```
Δ̂ = β(log πθ(yₘ|x) - log πθ(yₗ|x) - log πref(yₘ|x) + log πref(yₗ|x))
```

#### Advantages of DPO

1. **Simpler**: No reward model, no RL training loop
2. **Stable**: Direct supervised learning
3. **Efficient**: Single training phase
4. **Effective**: Comparable or better than PPO on many tasks

### 3. GRPO (Group Relative Policy Optimization)

**Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (DeepSeek-AI, 2024)](https://arxiv.org/abs/2402.03300)

GRPO is a variant that optimizes relative to group statistics, particularly effective for math and reasoning tasks.

#### Key Idea

Instead of using a learned value function, GRPO uses group-based advantages:

```
Âᵢ = r(xᵢ, yᵢ) - mean(r(xⱼ, yⱼ) for j in group)
```

#### GRPO Objective

```
L_GRPO(θ) = E[min(rₜ(θ)Âₜ^group, clip(rₜ(θ), 1-ε, 1+ε)Âₜ^group) - β·DKL(πθ || πref)]
```

Where the group advantage is:

```
Âᵢ^group = (rᵢ - μ_group) / (σ_group + δ)
```

With:
- **μ_group**: Mean reward in the group
- **σ_group**: Standard deviation of rewards in group
- **δ**: Small constant for numerical stability

#### Implementation Details

For each prompt x:
1. Sample G responses: y₁, y₂, ..., y_G
2. Compute rewards: r₁, r₂, ..., r_G
3. Normalize advantages within group:
   ```
   Âᵢ = (rᵢ - mean(r)) / (std(r) + 1e-8)
   ```
4. Update policy using PPO-style objective

#### Advantages of GRPO

1. **No value network**: Reduces memory and computation
2. **Better exploration**: Group normalization encourages diversity
3. **Effective for reasoning**: Strong results on math/code tasks
4. **Simpler training**: Fewer hyperparameters than PPO

### 4. REINFORCE

**Paper**: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696)

The foundational policy gradient algorithm.

#### REINFORCE Objective

```
∇θ J(θ) = E[∑ₜ ∇θ log πθ(aₜ|sₜ) · Rₜ]
```

Where Rₜ is the return from time t:

```
Rₜ = ∑(k=t to T) γ^(k-t) rₖ
```

#### With Baseline

To reduce variance, subtract a baseline b(sₜ):

```
∇θ J(θ) = E[∑ₜ ∇θ log πθ(aₜ|sₜ) · (Rₜ - b(sₜ))]
```

Common baselines:
- **Moving average**: b = running average of returns
- **Value function**: b(s) = V(s)

### 5. Additional Algorithms

#### RLHF with Rejection Sampling

**Paper**: [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

Instead of RL, use the reward model for rejection sampling:
1. Sample K responses for each prompt
2. Select the highest-reward response
3. Fine-tune on selected responses

#### RRHF (Rank Responses to align Human Feedback)

**Paper**: [RRHF: Rank Responses to Align Language Models with Human Feedback (Yuan et al., 2023)](https://arxiv.org/abs/2304.05302)

Ranking-based loss:

```
L_RRHF = -E[∑ᵢ (rᵢ/τ) · log πθ(yᵢ|x) / ∑ⱼ exp(rⱼ/τ)]
```

Where τ is a temperature parameter.

#### IPO (Identity Preference Optimization)

**Paper**: [A General Theoretical Paradigm to Understand Learning from Human Preferences (Azar et al., 2023)](https://arxiv.org/abs/2310.12036)

Addresses DPO overfitting with regularization:

```
L_IPO(θ) = E[(log πθ(yₘ|x)/πref(yₘ|x) - log πθ(yₗ|x)/πref(yₗ|x) - 1/2)²]
```

#### KTO (Kahneman-Tversky Optimization)

**Paper**: [KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306)

Based on prospect theory, works with binary feedback:

```
L_KTO = E[v_KTO(∇̂) · (1 - σ(∇̂))]
```

Where v_KTO is a value function inspired by prospect theory.

## Comparison of Algorithms

| Algorithm | Complexity | Sample Efficiency | Stability | Memory | Best For |
|-----------|-----------|-------------------|-----------|---------|----------|
| **PPO** | High | Medium | High | High (value network) | General RLHF |
| **DPO** | Low | High | Very High | Low | Preference data |
| **GRPO** | Medium | High | High | Medium | Math/Reasoning |
| **REINFORCE** | Low | Low | Low | Low | Simple tasks |
| **Rejection Sampling** | Very Low | Low | Very High | Low | Quick alignment |

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- CUDA Toolkit 11.8+ or 12.0+

### Installation

**Using uv (recommended):**

```bash
# Assuming you already have the quantization_tutorial repo
cd quantization_tutorial

# Install additional dependencies for RL
uv pip install trl transformers accelerate peft datasets wandb

# Verify setup
python -c "import trl; print(f'TRL version: {trl.__version__}')"
```

**Using pip:**

```bash
pip install trl transformers accelerate peft datasets wandb
```

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 16GB VRAM (RTX 3090, A5000)
- RAM: 32GB system RAM

**Recommended:**
- GPU: NVIDIA A100 (40GB or 80GB)
- RAM: 64GB+ system RAM

## Tutorial Examples

### Example 1: Basic REINFORCE for LLMs
**File:** `examples/rl_01_reinforce.py`

Learn the fundamentals of policy gradient methods applied to language models.

```bash
python examples/rl_01_reinforce.py
```

### Example 2: PPO Training
**File:** `examples/rl_02_ppo_training.py`

Implement full PPO training loop with KL penalty and value function.

```bash
python examples/rl_02_ppo_training.py
```

### Example 3: DPO Training
**File:** `examples/rl_03_dpo_training.py`

Train a model using Direct Preference Optimization.

```bash
python examples/rl_03_dpo_training.py
```

### Example 4: GRPO for Math Reasoning
**File:** `examples/rl_04_grpo_training.py`

Implement GRPO for mathematical reasoning tasks.

```bash
python examples/rl_04_grpo_training.py
```

### Example 5: Reward Model Training
**File:** `examples/rl_05_reward_modeling.py`

Train a reward model from preference data.

```bash
python examples/rl_05_reward_modeling.py
```

### Example 6: RLHF Full Pipeline
**File:** `examples/rl_06_rlhf_pipeline.py`

Complete RLHF pipeline: SFT → Reward Modeling → PPO.

```bash
python examples/rl_06_rlhf_pipeline.py
```

## Key Implementation Details

### Computing Advantages (GAE)

```python
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        values: List of value estimates [V(s_0), ..., V(s_T+1)]
        gamma: Discount factor
        lambda_: GAE parameter
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    return advantages
```

### KL Penalty Computation

```python
def kl_penalty(logprobs_policy, logprobs_ref):
    """
    Compute KL divergence between policy and reference

    KL(π || π_ref) = E[log π - log π_ref]
    """
    return (logprobs_policy - logprobs_ref).mean()
```

### PPO Clipped Objective

```python
def ppo_loss(logprobs, old_logprobs, advantages, clip_eps=0.2):
    """
    Compute PPO clipped surrogate objective
    """
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()
```

## Training Tips

### 1. Hyperparameters

**PPO:**
- Learning rate: 1e-6 to 5e-6
- Clip epsilon: 0.1 to 0.2
- KL penalty (β): 0.01 to 0.1
- GAE lambda: 0.95
- PPO epochs: 4
- Batch size: 16-64 prompts

**DPO:**
- Learning rate: 5e-7 to 5e-6
- Beta (β): 0.1 to 0.5
- Batch size: 32-128 pairs
- Epochs: 1-3

**GRPO:**
- Learning rate: 1e-6 to 5e-6
- Group size: 8-64
- Clip epsilon: 0.2
- KL penalty: 0.01 to 0.05

### 2. Common Issues

**High KL Divergence:**
- Increase β (KL penalty coefficient)
- Decrease learning rate
- Use smaller clip epsilon

**Low Reward:**
- Check reward model calibration
- Reduce KL penalty
- Increase sampling temperature

**Training Instability:**
- Use gradient clipping (max norm 1.0)
- Reduce learning rate
- Use warmup schedule

**Memory Issues:**
- Use gradient checkpointing
- Reduce batch size
- Use LORA/QLora for parameter-efficient training

## Advanced Topics

### Parameter-Efficient RLHF with LoRA

Instead of fine-tuning all parameters, use Low-Rank Adaptation:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
```

### Constitutional AI

**Paper**: [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)

Use AI-generated feedback instead of human feedback:
1. Generate responses
2. Use an AI critic to evaluate against principles
3. Fine-tune on AI preferences

### Iterative RLHF

Iterative refinement:
1. RLHF iteration 1 → Model v1
2. Collect new preferences using v1
3. RLHF iteration 2 → Model v2
4. Repeat

## Performance Metrics

### During Training

1. **Average Reward**: Track improvement over time
2. **KL Divergence**: Monitor divergence from reference
3. **Policy Entropy**: Ensure exploration isn't too low
4. **Explained Variance**: Value function quality (for PPO)

### Evaluation

1. **Win Rate**: Human preference evaluations
2. **GPT-4 as Judge**: Automated evaluation
3. **Task-Specific Metrics**:
   - MMLU (knowledge)
   - HumanEval (coding)
   - GSM8K (math)
4. **Safety Metrics**: Toxicity, bias, jailbreak resistance

## References

### Foundational Papers

1. **REINFORCE**: Williams, R. J. (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). Machine Learning.

2. **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv.

3. **GAE**: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). arXiv.

### RLHF Papers

4. **InstructGPT**: Ouyang, L., et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS.

5. **Anthropic RLHF**: Bai, Y., et al. (2022). [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). arXiv.

6. **Constitutional AI**: Bai, Y., et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). arXiv.

### Modern Algorithms

7. **DPO**: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). NeurIPS.

8. **GRPO/DeepSeekMath**: Shao, Z., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv.

9. **RRHF**: Yuan, Z., et al. (2023). [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302). arXiv.

10. **IPO**: Azar, M. G., et al. (2023). [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036). arXiv.

11. **KTO**: Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306). arXiv.

### Reward Modeling

12. **Reward Modeling**: Stiennon, N., et al. (2020). [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325). NeurIPS.

13. **Scaling Laws for Reward Models**: Gao, L., et al. (2022). [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). arXiv.

### Safety & Alignment

14. **Red Teaming**: Ganguli, D., et al. (2022). [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858). arXiv.

15. **RLHF Limitations**: Casper, S., et al. (2023). [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217). arXiv.

### Survey Papers

16. **Alignment Survey**: Ji, J., et al. (2023). [AI Alignment: A Comprehensive Survey](https://arxiv.org/abs/2310.19852). arXiv.

17. **RLHF Survey**: Kaufmann, T., et al. (2024). [A Survey on Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925). arXiv.

### Libraries & Tools

- **TRL (Transformer Reinforcement Learning)**: https://github.com/huggingface/trl
- **OpenRLHF**: https://github.com/OpenLLMAI/OpenRLHF
- **DeepSpeed-Chat**: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
- **Anthropic's Preference Model Pretraining**: https://github.com/anthropics/hh-rlhf

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details

---

**Happy Training! 🚀**
