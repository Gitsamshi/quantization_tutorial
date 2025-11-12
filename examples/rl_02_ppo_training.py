"""
Example 2: PPO (Proximal Policy Optimization) Training for LLMs

This example demonstrates how to implement PPO for RLHF training of language models.

Mathematical Background:
========================

PPO Objective (Clipped):
    L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
    - r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)  [probability ratio]
    - Â_t: Advantage estimate (GAE)
    - ε: Clipping parameter (typically 0.2)

Full PPO Loss for LLMs:
    L(θ) = L^CLIP(θ) - β·D_KL(π_θ || π_ref) + c_1·L^VF(θ)

Generalized Advantage Estimation (GAE):
    Â_t = Σ(γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

References:
    - PPO: https://arxiv.org/abs/1707.06347 (Schulman et al., 2017)
    - GAE: https://arxiv.org/abs/1506.02438 (Schulman et al., 2015)
    - InstructGPT: https://arxiv.org/abs/2203.02155 (Ouyang et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import math

print("=" * 80)
print("PPO Training for LLMs")
print("=" * 80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

@dataclass
class PPOConfig:
    """PPO Configuration"""
    # Model settings
    model_name: str = "gpt2"

    # PPO hyperparameters
    learning_rate: float = 1e-5
    clip_eps: float = 0.2  # ε in the paper
    value_coef: float = 0.1  # c_1 in the paper
    kl_penalty: float = 0.02  # β in the paper
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # λ for GAE

    # Training settings
    ppo_epochs: int = 4
    batch_size: int = 4
    max_length: int = 128
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Generation settings
    num_return_sequences: int = 4
    temperature: float = 0.7


class RolloutBuffer:
    """
    Buffer to store rollout data for PPO training.

    Stores sequences, actions (tokens), log probabilities, rewards, values, etc.
    """
    def __init__(self):
        self.sequences = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.ref_logprobs = []

    def add(self, sequence, logprob, reward, value, ref_logprob):
        self.sequences.append(sequence)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.ref_logprobs.append(ref_logprob)

    def compute_gae(self, gamma: float, gae_lambda: float):
        """
        Compute Generalized Advantage Estimation.

        Mathematical formula:
            Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

        Where TD error:
            δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            gamma: Discount factor
            gae_lambda: GAE parameter λ
        """
        advantages = []
        returns = []

        for rewards, values in zip(self.rewards, self.values):
            # Convert to tensors
            rewards = torch.tensor(rewards, dtype=torch.float32)
            values = torch.tensor(values + [0], dtype=torch.float32)  # Add bootstrap value

            # Compute advantages using GAE
            gae = 0
            advantages_seq = []

            for t in reversed(range(len(rewards))):
                # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
                delta = rewards[t] + gamma * values[t+1] - values[t]

                # GAE recursion: Â_t = δ_t + (γλ)Â_{t+1}
                gae = delta + gamma * gae_lambda * gae
                advantages_seq.insert(0, gae)

            advantages_seq = torch.tensor(advantages_seq)

            # Returns = Advantages + Values
            returns_seq = advantages_seq + values[:-1]

            advantages.append(advantages_seq)
            returns.append(returns_seq)

        self.advantages = advantages
        self.returns = returns

    def get(self):
        """Get all data from buffer."""
        return {
            'sequences': self.sequences,
            'logprobs': self.logprobs,
            'advantages': self.advantages,
            'returns': self.returns,
            'ref_logprobs': self.ref_logprobs
        }

    def clear(self):
        """Clear the buffer."""
        self.__init__()


class ValueNetwork(nn.Module):
    """
    Value function V(s) for PPO.

    Estimates expected return from state s.
    We use the same base model with a value head.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass to compute value estimates.

        Returns:
            values: V(s) for each position
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]

        # Compute values
        values = self.value_head(hidden_states).squeeze(-1)

        return values


def compute_log_probs(model, input_ids, attention_mask=None):
    """
    Compute log probabilities for each token in the sequence.

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]

    Returns:
        log_probs: Log probability of each token [batch_size, seq_len]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Select log probs of actual tokens
        # Shift by 1 (predict next token)
        log_probs = torch.gather(
            log_probs_all[:, :-1, :],
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    return log_probs


def compute_rewards(reward_model, prompts, responses, tokenizer):
    """
    Compute rewards using a reward model.

    For this example, we use a simple heuristic reward:
    - Length penalty (prefer concise answers)
    - Sentiment (prefer positive responses)

    In practice, you would train a reward model on human preferences.

    Args:
        reward_model: Reward model (or None for heuristic)
        prompts: List of prompt texts
        responses: List of response texts

    Returns:
        rewards: List of reward values
    """
    rewards = []

    for prompt, response in zip(prompts, responses):
        # Simple heuristic reward for demonstration
        # In practice, use a trained reward model

        # Length penalty (prefer 10-50 tokens)
        response_length = len(tokenizer.encode(response))
        length_penalty = -abs(response_length - 30) / 30.0

        # Keyword rewards (simple sentiment)
        positive_words = ['good', 'great', 'excellent', 'helpful', 'thank']
        negative_words = ['bad', 'poor', 'terrible', 'wrong', 'error']

        response_lower = response.lower()
        sentiment_reward = sum(1 for w in positive_words if w in response_lower)
        sentiment_reward -= sum(1 for w in negative_words if w in response_lower)

        total_reward = length_penalty + 0.5 * sentiment_reward
        rewards.append(total_reward)

    return rewards


def ppo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO clipped surrogate objective.

    Mathematical formula:
        L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    Where:
        r_t(θ) = exp(log π_θ - log π_old) = π_θ / π_old

    Args:
        policy_logprobs: Log probs from current policy
        old_logprobs: Log probs from old policy
        advantages: Advantage estimates
        clip_eps: Clipping parameter ε

    Returns:
        loss: PPO clipped loss
    """
    # Probability ratio: r_t(θ) = π_θ / π_old
    ratio = torch.exp(policy_logprobs - old_logprobs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # PPO objective (we want to maximize, so negate for loss)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss.mean()


def value_loss(predicted_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Compute value function loss (MSE).

    L^VF = E[(V_θ(s) - R)^2]

    Args:
        predicted_values: V(s) from value network
        returns: Actual returns

    Returns:
        loss: MSE loss
    """
    return F.mse_loss(predicted_values, returns)


def kl_divergence(logprobs_policy: torch.Tensor, logprobs_ref: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference.

    Mathematical formula:
        D_KL(π || π_ref) = E[log π - log π_ref]

    Args:
        logprobs_policy: Log probs from current policy
        logprobs_ref: Log probs from reference policy

    Returns:
        kl: KL divergence
    """
    return (logprobs_policy - logprobs_ref).mean()


def train_ppo_step(
    policy_model,
    value_model,
    ref_model,
    optimizer_policy,
    optimizer_value,
    buffer: RolloutBuffer,
    config: PPOConfig
):
    """
    Perform one PPO training step.

    Full PPO loss:
        L(θ) = L^CLIP(θ) - β·D_KL(π_θ || π_ref) + c_1·L^VF(θ)

    Args:
        policy_model: Current policy (being trained)
        value_model: Value function network
        ref_model: Reference policy (frozen)
        optimizer_policy: Optimizer for policy
        optimizer_value: Optimizer for value function
        buffer: Rollout buffer with experiences
        config: PPO configuration
    """
    data = buffer.get()

    total_policy_loss = 0
    total_value_loss = 0
    total_kl = 0

    # PPO epochs: reuse data multiple times
    for epoch in range(config.ppo_epochs):
        for i in range(len(data['sequences'])):
            # Get data for this sequence
            input_ids = data['sequences'][i]
            old_logprobs = data['logprobs'][i]
            advantages = data['advantages'][i]
            returns = data['returns'][i]
            ref_logprobs = data['ref_logprobs'][i]

            # Normalize advantages (important for stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            # Compute PPO loss
            pg_loss = ppo_loss(policy_logprobs, old_logprobs, advantages, config.clip_eps)

            # Compute KL penalty
            kl = kl_divergence(policy_logprobs, ref_logprobs)

            # Total policy loss
            policy_loss = pg_loss + config.kl_penalty * kl

            # Backward pass
            policy_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)

            # Update policy
            optimizer_policy.step()
            optimizer_policy.zero_grad()

            # Train value function
            values = value_model(input_ids=input_ids.unsqueeze(0))
            values = values.squeeze()[:-1]  # Match sequence length

            vf_loss = value_loss(values, returns)

            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), config.max_grad_norm)
            optimizer_value.step()
            optimizer_value.zero_grad()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += vf_loss.item()
            total_kl += kl.item()

    num_updates = len(data['sequences']) * config.ppo_epochs

    return {
        'policy_loss': total_policy_loss / num_updates,
        'value_loss': total_value_loss / num_updates,
        'kl_divergence': total_kl / num_updates
    }


def main():
    """Main training loop."""
    config = PPOConfig()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Clip Epsilon (ε): {config.clip_eps}")
    print(f"KL Penalty (β): {config.kl_penalty}")
    print(f"GAE Lambda (λ): {config.gae_lambda}")
    print(f"PPO Epochs: {config.ppo_epochs}")

    # Load tokenizer and models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (being trained)
    policy_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)

    # Reference model (frozen, for KL penalty)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Value model
    base_model_for_value = AutoModelForCausalLM.from_pretrained(config.model_name)
    value_model = ValueNetwork(base_model_for_value).to(device)

    print("✓ Models loaded")

    # Optimizers
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=config.learning_rate)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=config.learning_rate)

    # Example prompts
    prompts = [
        "Explain what reinforcement learning is:",
        "What is the capital of France?",
        "Write a short poem about AI:",
        "How does photosynthesis work?"
    ]

    print("\n" + "=" * 80)
    print("PPO Training Loop")
    print("=" * 80)

    num_iterations = 3

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*80}")

        # Rollout buffer
        buffer = RolloutBuffer()

        # Generate responses and collect data
        policy_model.eval()

        for prompt in prompts:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            # Generate response
            with torch.no_grad():
                outputs = policy_model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_return_sequences=1,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            full_sequence = outputs[0]
            response_text = tokenizer.decode(full_sequence, skip_special_tokens=True)

            # Compute log probs from current policy
            logprobs = compute_log_probs(policy_model, full_sequence.unsqueeze(0))

            # Compute log probs from reference policy
            ref_logprobs = compute_log_probs(ref_model, full_sequence.unsqueeze(0))

            # Compute values
            values = value_model(full_sequence.unsqueeze(0)).squeeze().tolist()

            # Compute reward
            prompt_only = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            response_only = response_text[len(prompt_only):]

            rewards = compute_rewards(None, [prompt_only], [response_only], tokenizer)

            # For simplicity, assign same reward to all tokens in response
            reward_per_token = [rewards[0]] * len(logprobs.squeeze())

            # Add to buffer
            buffer.add(
                sequence=full_sequence,
                logprob=logprobs.squeeze(),
                reward=reward_per_token,
                value=values,
                ref_logprob=ref_logprobs.squeeze()
            )

        # Compute advantages using GAE
        buffer.compute_gae(config.gamma, config.gae_lambda)

        # Train on collected data
        policy_model.train()
        metrics = train_ppo_step(
            policy_model,
            value_model,
            ref_model,
            optimizer_policy,
            optimizer_value,
            buffer,
            config
        )

        print(f"\nMetrics:")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")

        # Sample generation
        if iteration % 1 == 0:
            print(f"\nSample Generation:")
            test_prompt = "What is machine learning?"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = policy_model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Prompt: {test_prompt}")
            print(f"  Response: {response}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. PPO uses clipped objective to prevent large policy updates")
    print("2. GAE computes advantages efficiently with bias-variance trade-off")
    print("3. KL penalty keeps policy close to reference (prevents mode collapse)")
    print("4. Value function learns to estimate returns (reduces variance)")
    print("\nNext steps:")
    print("- Use a trained reward model instead of heuristic rewards")
    print("- Scale up to larger models with LoRA/QLoRA")
    print("- Use proper datasets (e.g., Anthropic HH-RLHF)")
    print("- Track metrics with W&B or TensorBoard")


if __name__ == "__main__":
    main()
