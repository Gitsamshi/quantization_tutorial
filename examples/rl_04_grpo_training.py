"""
Example 4: GRPO (Group Relative Policy Optimization) Training

This example demonstrates how to implement GRPO, a variant of PPO that is
particularly effective for mathematical reasoning and code generation tasks.

Mathematical Background:
========================

GRPO modifies PPO by using group-based normalization instead of a learned value function.

Key Idea:
    For each prompt x, sample G responses: y1, y2, ..., yG
    Compute rewards: r1, r2, ..., rG
    Normalize advantages within group (not using a value network):

        Â_i^group = (r_i - μ_group) / (σ_group + δ)

    Where:
        μ_group = mean(r1, ..., rG)
        σ_group = std(r1, ..., rG)
        δ = small constant for stability (e.g., 1e-8)

GRPO Objective:
    L_GRPO(θ) = E[min(r_t(θ)Â_t^group, clip(r_t(θ), 1-ε, 1+ε)Â_t^group) - β·D_KL(π_θ || π_ref)]

Advantages over PPO:
    1. No value network → reduced memory and computation
    2. Group normalization → better exploration
    3. Simpler training → fewer hyperparameters
    4. Effective for reasoning tasks

Algorithm Steps:
    1. For each prompt x in batch:
       a. Sample G responses from current policy
       b. Compute reward for each response
    2. Within each group, normalize rewards:
       Â_i = (r_i - mean(r)) / (std(r) + 1e-8)
    3. Update policy using PPO-style objective with group advantages
    4. Apply KL penalty to prevent divergence from reference

References:
    - DeepSeekMath: https://arxiv.org/abs/2402.03300 (DeepSeek-AI, 2024)
    - PPO: https://arxiv.org/abs/1707.06347 (Schulman et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import re

print("=" * 80)
print("GRPO (Group Relative Policy Optimization) Training")
print("=" * 80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


@dataclass
class GRPOConfig:
    """GRPO Configuration"""
    # Model settings
    model_name: str = "gpt2"

    # GRPO hyperparameters
    learning_rate: float = 1e-5
    clip_eps: float = 0.2  # ε for clipping
    kl_penalty: float = 0.01  # β for KL penalty
    group_size: int = 8  # G - number of samples per prompt

    # Training settings
    ppo_epochs: int = 4
    batch_size: int = 4
    max_length: int = 256
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    # Generation settings
    temperature: float = 0.8
    top_p: float = 0.95

    # Task-specific
    task: str = "math"  # 'math' or 'general'


class MathDataset(Dataset):
    """
    Simple math problems dataset.

    GRPO is particularly effective for math reasoning tasks.
    """
    def __init__(self):
        self.problems = [
            {
                'question': 'What is 15 + 27?',
                'answer': '42'
            },
            {
                'question': 'Calculate 8 × 9',
                'answer': '72'
            },
            {
                'question': 'What is 100 - 37?',
                'answer': '63'
            },
            {
                'question': 'Solve: 3 + 4 × 5',
                'answer': '23'
            },
            {
                'question': 'What is 144 ÷ 12?',
                'answer': '12'
            },
            {
                'question': 'Calculate 2^5',
                'answer': '32'
            }
        ]

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


class GroupRolloutBuffer:
    """
    Buffer for storing group-based rollouts.

    For each prompt, stores G responses with their rewards.
    """
    def __init__(self):
        self.groups = []  # List of groups, each group is a dict

    def add_group(self, prompt: str, responses: List[Dict]):
        """
        Add a group of responses for a single prompt.

        Args:
            prompt: The input prompt
            responses: List of response dicts with keys:
                - 'sequence': Full token sequence
                - 'logprobs': Log probabilities
                - 'reward': Reward value
                - 'ref_logprobs': Reference log probs
        """
        self.groups.append({
            'prompt': prompt,
            'responses': responses
        })

    def compute_group_advantages(self):
        """
        Compute advantages using group normalization.

        Mathematical formula:
            Â_i = (r_i - μ) / (σ + δ)

        Where:
            μ = mean of rewards in group
            σ = standard deviation of rewards in group
            δ = small constant (1e-8)

        This normalization:
            1. Centers advantages around 0
            2. Scales by group statistics
            3. Encourages diversity (different responses get different advantages)
        """
        for group in self.groups:
            rewards = [r['reward'] for r in group['responses']]
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # Group statistics
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std()

            # Normalize advantages
            for i, response in enumerate(group['responses']):
                advantage = (rewards[i] - mean_reward) / (std_reward + 1e-8)
                response['advantage'] = advantage

    def get_all_samples(self):
        """Get all samples with their advantages."""
        samples = []
        for group in self.groups:
            for response in group['responses']:
                samples.append(response)
        return samples

    def clear(self):
        """Clear the buffer."""
        self.groups = []


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from generated text.

    Args:
        text: Generated text

    Returns:
        answer: Extracted answer string
    """
    # Simple heuristic: find last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""


def compute_math_reward(question: str, generated_text: str, correct_answer: str) -> float:
    """
    Compute reward for math problem.

    Reward components:
        1. Correctness (main signal): +10 if correct, -1 if wrong
        2. Length penalty: Prefer concise solutions
        3. Format bonus: Bonus for showing work

    Args:
        question: Math question
        generated_text: Generated response
        correct_answer: Correct answer

    Returns:
        reward: Total reward value
    """
    # Extract answer from generation
    predicted_answer = extract_answer(generated_text)

    # Correctness reward
    if predicted_answer.strip() == correct_answer.strip():
        correctness_reward = 10.0
    else:
        correctness_reward = -1.0

    # Length penalty (prefer solutions between 20-100 tokens)
    length = len(generated_text.split())
    if 20 <= length <= 100:
        length_reward = 0.0
    else:
        length_reward = -abs(length - 60) / 100.0

    # Format bonus (check if shows reasoning)
    format_bonus = 0.0
    if any(keyword in generated_text.lower() for keyword in ['because', 'therefore', 'so', 'thus', '=']):
        format_bonus = 0.5

    total_reward = correctness_reward + length_reward + format_bonus

    return total_reward


def compute_log_probs(model, input_ids, attention_mask=None):
    """
    Compute log probabilities for generated tokens.

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask

    Returns:
        log_probs: Sum of log probabilities
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Get log probs of actual tokens (shifted by 1)
        log_probs = torch.gather(
            log_probs_all[:, :-1, :],
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Sum over sequence
        total_log_prob = log_probs.sum(dim=-1)

    return total_log_prob


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2
) -> torch.Tensor:
    """
    Compute GRPO loss (same as PPO clipped objective).

    Mathematical formula:
        L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    Where:
        r_t(θ) = π_θ / π_old = exp(log π_θ - log π_old)

    Args:
        policy_logprobs: Log probs from current policy
        old_logprobs: Log probs from old policy
        advantages: Group-normalized advantages
        clip_eps: Clipping parameter

    Returns:
        loss: GRPO clipped loss
    """
    # Probability ratio
    ratio = torch.exp(policy_logprobs - old_logprobs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # GRPO objective (maximize, so negate for loss)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss.mean()


def train_grpo_step(
    policy_model,
    ref_model,
    optimizer,
    buffer: GroupRolloutBuffer,
    config: GRPOConfig
):
    """
    Perform one GRPO training step.

    Full GRPO loss:
        L(θ) = L^CLIP(θ) - β·D_KL(π_θ || π_ref)

    Args:
        policy_model: Current policy (being trained)
        ref_model: Reference policy (frozen)
        optimizer: Optimizer
        buffer: Rollout buffer with group data
        config: GRPO configuration
    """
    # Get all samples with computed advantages
    samples = buffer.get_all_samples()

    total_policy_loss = 0
    total_kl = 0
    num_updates = 0

    # PPO epochs: reuse data multiple times
    for epoch in range(config.ppo_epochs):
        for sample in samples:
            input_ids = sample['sequence'].to(device)
            old_logprobs = sample['logprobs'].to(device)
            advantage = torch.tensor(sample['advantage'], dtype=torch.float32).to(device)
            ref_logprobs = sample['ref_logprobs'].to(device)

            # Forward pass through policy
            outputs = policy_model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits

            # Compute log probs
            log_probs_all = F.log_softmax(logits, dim=-1)
            policy_logprobs = torch.gather(
                log_probs_all[:, :-1, :],
                dim=-1,
                index=input_ids[1:].unsqueeze(0).unsqueeze(-1)
            ).squeeze()

            # Sum log probs
            policy_logprobs_sum = policy_logprobs.sum()
            old_logprobs_sum = old_logprobs.sum() if old_logprobs.dim() > 0 else old_logprobs
            ref_logprobs_sum = ref_logprobs.sum() if ref_logprobs.dim() > 0 else ref_logprobs

            # Compute GRPO loss
            pg_loss = grpo_loss(
                policy_logprobs_sum.unsqueeze(0),
                old_logprobs_sum.unsqueeze(0),
                advantage.unsqueeze(0),
                config.clip_eps
            )

            # Compute KL penalty
            kl = (policy_logprobs_sum - ref_logprobs_sum).abs()

            # Total loss
            loss = pg_loss + config.kl_penalty * kl

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)

            # Update
            optimizer.step()
            optimizer.zero_grad()

            # Track metrics
            total_policy_loss += loss.item()
            total_kl += kl.item()
            num_updates += 1

    return {
        'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
        'kl_divergence': total_kl / num_updates if num_updates > 0 else 0
    }


def main():
    """Main training loop."""
    config = GRPOConfig()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Clip Epsilon (ε): {config.clip_eps}")
    print(f"KL Penalty (β): {config.kl_penalty}")
    print(f"Group Size (G): {config.group_size}")
    print(f"PPO Epochs: {config.ppo_epochs}")

    # Load tokenizer and models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (being trained)
    policy_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("✓ Models loaded")

    # Optimizer
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.learning_rate)

    # Dataset
    dataset = MathDataset()
    print(f"✓ Dataset loaded: {len(dataset)} math problems")

    # Training loop
    print("\n" + "=" * 80)
    print("GRPO Training Loop")
    print("=" * 80)

    num_iterations = 5

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*80}")

        # Buffer for this iteration
        buffer = GroupRolloutBuffer()

        # Generate groups for each problem
        policy_model.eval()

        for problem_idx, problem in enumerate(dataset):
            question = problem['question']
            correct_answer = problem['answer']

            prompt = f"Question: {question}\nAnswer: "

            print(f"\nProblem {problem_idx + 1}: {question}")

            # Sample G responses for this prompt
            group_responses = []

            for g in range(config.group_size):
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Generate
                with torch.no_grad():
                    outputs = policy_model.generate(
                        **inputs,
                        max_length=config.max_length,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode
                full_sequence = outputs[0]
                full_text = tokenizer.decode(full_sequence, skip_special_tokens=True)
                response_text = full_text[len(prompt):]

                # Compute log probs
                logprobs = compute_log_probs(policy_model, full_sequence.unsqueeze(0))
                ref_logprobs = compute_log_probs(ref_model, full_sequence.unsqueeze(0))

                # Compute reward
                reward = compute_math_reward(question, response_text, correct_answer)

                # Store in group
                group_responses.append({
                    'sequence': full_sequence,
                    'logprobs': logprobs.squeeze(),
                    'reward': reward,
                    'ref_logprobs': ref_logprobs.squeeze(),
                    'text': response_text
                })

            # Add group to buffer
            buffer.add_group(prompt, group_responses)

            # Show sample responses
            print(f"  Sample responses (Group size: {config.group_size}):")
            for i, resp in enumerate(group_responses[:3]):  # Show first 3
                print(f"    [{i+1}] Reward: {resp['reward']:+.2f} | {resp['text'][:80]}")

        # Compute group advantages
        buffer.compute_group_advantages()

        # Train on collected data
        policy_model.train()
        metrics = train_grpo_step(
            policy_model,
            ref_model,
            optimizer,
            buffer,
            config
        )

        print(f"\nMetrics:")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")

        # Evaluation
        if iteration % 1 == 0:
            print(f"\nEvaluation:")
            policy_model.eval()

            correct = 0
            total = 0

            for problem in dataset:
                question = problem['question']
                correct_answer = problem['answer']
                prompt = f"Question: {question}\nAnswer: "

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = policy_model.generate(
                        **inputs,
                        max_length=config.max_length,
                        temperature=0.1,  # Low temperature for evaluation
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = response[len(prompt):]

                predicted = extract_answer(response_text)

                if predicted.strip() == correct_answer.strip():
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            print(f"  Accuracy: {correct}/{total} = {accuracy:.2%}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Mathematical Summary
    print("\n" + "=" * 80)
    print("Mathematical Summary")
    print("=" * 80)
    print("""
GRPO Algorithm:
    1. For each prompt x, sample G responses: y1, ..., yG
    2. Compute rewards: r1, ..., rG
    3. Normalize within group:
           Âi = (ri - μ) / (σ + δ)
       where μ = mean(r), σ = std(r)
    4. Update policy using PPO objective with group advantages

GRPO Loss:
    L(θ) = L^CLIP(θ) - β·D_KL(π_θ || π_ref)

Where:
    L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
    r_t(θ) = π_θ / π_old

Key Differences from PPO:
    ✓ No value network (use group statistics instead)
    ✓ Group normalization (better exploration)
    ✓ Simpler architecture (fewer parameters)
    ✓ Memory efficient (no value network gradients)

When to Use GRPO:
    ✓ Math reasoning tasks
    ✓ Code generation
    ✓ Tasks with verifiable rewards
    ✓ When you want simpler training than PPO

Hyperparameters:
    - Group size (G): Typically 8-64
    - Clip epsilon (ε): 0.2 works well
    - KL penalty (β): 0.01-0.05 (lower than PPO)
    """)

    print("\nNext steps:")
    print("- Try larger group sizes for better advantage estimates")
    print("- Use real math datasets (GSM8K, MATH)")
    print("- Implement process reward models (reward intermediate steps)")
    print("- Compare with PPO and DPO on same tasks")


if __name__ == "__main__":
    main()
