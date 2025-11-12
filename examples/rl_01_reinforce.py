"""
Example 1: REINFORCE Algorithm for LLMs

This example demonstrates the foundational REINFORCE policy gradient algorithm
applied to language models. REINFORCE is the simplest policy gradient method
and forms the basis for more advanced algorithms like PPO.

Mathematical Background:
========================

Policy Gradient Theorem:
    ∇θ J(θ) = E[∑t ∇θ log πθ(at|st) · Rt]

Where:
    - J(θ): Expected return (objective to maximize)
    - πθ: Policy (language model) with parameters θ
    - at: Action (token) at time t
    - st: State (prompt + tokens generated so far)
    - Rt: Return (cumulative reward from time t)

For language generation (episodic, terminal reward only):
    Rt = R  (same for all t in sequence)

REINFORCE Update:
    θ ← θ + α · ∇θ log πθ(y|x) · R

Where:
    - x: Prompt
    - y: Generated sequence
    - R: Total reward for sequence
    - α: Learning rate

Variance Reduction with Baseline:
    ∇θ J(θ) = E[∑t ∇θ log πθ(at|st) · (Rt - b(st))]

Where b(st) is a baseline (e.g., moving average of returns).
This reduces variance without introducing bias.

References:
    - REINFORCE: Williams, 1992 (https://link.springer.com/article/10.1007/BF00992696)
    - Policy Gradients: Sutton & Barto, "Reinforcement Learning: An Introduction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
from dataclasses import dataclass

print("=" * 80)
print("REINFORCE Algorithm for LLMs")
print("=" * 80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


@dataclass
class REINFORCEConfig:
    """REINFORCE Configuration"""
    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 100
    temperature: float = 0.8
    num_iterations: int = 10
    baseline_decay: float = 0.9  # For exponential moving average baseline


def compute_sequence_log_prob(model, input_ids, attention_mask=None):
    """
    Compute log probability of a sequence.

    log π(y|x) = ∑t log π(yt|y<t, x)

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]

    Returns:
        log_probs: Total log probability for each sequence [batch_size]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Compute log probabilities for each token
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Get log probs of actual next tokens
    # Shape: [batch_size, seq_len-1]
    log_probs = torch.gather(
        log_probs_all[:, :-1, :],
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # Sum over sequence
    total_log_prob = log_probs.sum(dim=-1)

    return total_log_prob


def compute_simple_reward(prompt: str, response: str, tokenizer) -> float:
    """
    Compute a simple heuristic reward.

    For demonstration purposes. In practice, use:
    - Trained reward model
    - Human feedback
    - Task-specific metrics

    Args:
        prompt: Input prompt
        response: Generated response
        tokenizer: Tokenizer

    Returns:
        reward: Scalar reward value
    """
    # Length reward (prefer moderate length)
    response_tokens = tokenizer.encode(response)
    ideal_length = 30
    length_reward = -abs(len(response_tokens) - ideal_length) / ideal_length

    # Content rewards
    response_lower = response.lower()

    # Positive words
    positive_words = ['good', 'great', 'excellent', 'helpful', 'yes', 'correct']
    positive_score = sum(1 for word in positive_words if word in response_lower)

    # Negative words
    negative_words = ['bad', 'terrible', 'wrong', 'no', 'error']
    negative_score = sum(1 for word in negative_words if word in response_lower)

    content_reward = positive_score - negative_score

    # Coherence (simple heuristic: presence of common words)
    coherence_reward = 0.5 if any(word in response_lower for word in ['the', 'is', 'are', 'to']) else 0

    total_reward = length_reward + 0.5 * content_reward + coherence_reward

    return float(total_reward)


def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor, baseline: float = 0.0) -> torch.Tensor:
    """
    Compute REINFORCE loss.

    Mathematical formula:
        L = -E[log π(y|x) · (R - b)]

    We minimize negative expected return (maximize return).

    Args:
        log_probs: Log probabilities of generated sequences
        rewards: Rewards for sequences
        baseline: Baseline for variance reduction

    Returns:
        loss: REINFORCE loss
    """
    # Advantage: A = R - b
    advantages = rewards - baseline

    # REINFORCE loss: -E[log π · A]
    # We want to maximize log π when A > 0, minimize when A < 0
    loss = -(log_probs * advantages).mean()

    return loss


def main():
    """Main training loop."""
    config = REINFORCEConfig()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Baseline Decay: {config.baseline_decay}")

    # Load model and tokenizer
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)

    print("✓ Model loaded")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Example prompts
    prompts = [
        "Explain what is machine learning:",
        "What is the capital of France?",
        "Write a short poem:",
        "How does the internet work?"
    ]

    # Baseline (exponential moving average of rewards)
    baseline = 0.0

    print("\n" + "=" * 80)
    print("REINFORCE Training Loop")
    print("=" * 80)

    for iteration in range(config.num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'='*80}")

        # Lists to store batch data
        batch_log_probs = []
        batch_rewards = []

        # Generate samples
        model.eval()

        for prompt in prompts:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            full_sequence = outputs[0]
            full_text = tokenizer.decode(full_sequence, skip_special_tokens=True)

            # Extract response (remove prompt)
            response_text = full_text[len(prompt):]

            # Compute reward
            reward = compute_simple_reward(prompt, response_text, tokenizer)

            # Compute log probability of sequence
            log_prob = compute_sequence_log_prob(model, full_sequence.unsqueeze(0))

            batch_log_probs.append(log_prob.squeeze())
            batch_rewards.append(reward)

            print(f"\nPrompt: {prompt}")
            print(f"Response: {response_text[:100]}...")
            print(f"Reward: {reward:.4f}")

        # Convert to tensors
        batch_log_probs = torch.stack(batch_log_probs)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)

        # Update baseline (exponential moving average)
        current_avg_reward = batch_rewards.mean().item()
        baseline = config.baseline_decay * baseline + (1 - config.baseline_decay) * current_avg_reward

        print(f"\nAverage Reward: {current_avg_reward:.4f}")
        print(f"Baseline: {baseline:.4f}")

        # Training step
        model.train()

        # Compute REINFORCE loss
        loss = reinforce_loss(batch_log_probs, batch_rewards, baseline)

        print(f"Loss: {loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        print(f"\nUpdate applied")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    model.eval()

    test_prompts = [
        "What is artificial intelligence?",
        "Explain photosynthesis:",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

    # Mathematical Summary
    print("\n" + "=" * 80)
    print("Mathematical Summary")
    print("=" * 80)
    print("""
REINFORCE Algorithm:
    1. Sample trajectory: y ~ πθ(·|x)
    2. Compute return: R = reward(x, y)
    3. Compute gradient: ∇θ J = ∇θ log πθ(y|x) · (R - b)
    4. Update parameters: θ ← θ + α∇θ J

Policy Gradient Theorem:
    ∇θ J(θ) = E[∇θ log πθ(y|x) · R]

With Baseline (variance reduction):
    ∇θ J(θ) = E[∇θ log πθ(y|x) · (R - b)]

Key Properties:
    ✓ Unbiased gradient estimate
    ✓ Works for discrete actions (tokens)
    ✓ No critic/value function needed
    ✗ High variance (slow learning)
    ✗ Sample inefficient

Intuition:
    - If R > b: Increase probability of sequence y
    - If R < b: Decrease probability of sequence y
    - Magnitude of update proportional to (R - b)

Advantages:
    + Simple to implement
    + Theoretically sound
    + Forms basis for advanced methods (PPO, etc.)

Disadvantages:
    - High variance → slow convergence
    - Sample inefficient (each sample used once)
    - No explicit value function

Modern Improvements:
    → PPO: Add clipped objective, value function
    → A2C/A3C: Advantage estimation with critic
    → GRPO: Group-based normalization
    → DPO: Bypass RL entirely for preferences
    """)

    print("\nNext steps:")
    print("- Try different baseline strategies (learned value function)")
    print("- Implement multiple samples per prompt (reduce variance)")
    print("- Use advantage estimation (like in PPO)")
    print("- Apply to real tasks with trained reward models")
    print("- Compare with PPO and DPO")


if __name__ == "__main__":
    main()
