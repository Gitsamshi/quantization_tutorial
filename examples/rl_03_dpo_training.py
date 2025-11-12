"""
Example 3: DPO (Direct Preference Optimization) Training

This example demonstrates how to implement DPO for aligning language models
with human preferences WITHOUT explicit reward modeling or RL.

Mathematical Background:
========================

Key Insight:
Under the Bradley-Terry preference model, the optimal policy has a closed form:
    π*(y|x) = (1/Z(x)) · πref(y|x) · exp(r(x,y)/β)

Reparameterization (solving for reward):
    r(x, y) = β log(π*(y|x)/πref(y|x)) + β log Z(x)

DPO Objective:
Given preference data (x, yw, yl) where yw ≻ yl (yw preferred over yl):

    L_DPO(θ) = -E[log σ(β log(πθ(yw|x)/πref(yw|x)) - β log(πθ(yl|x)/πref(yl|x)))]

Where σ is the sigmoid function.

Simplified:
    L_DPO(θ) = -E[log σ(β Δlog π)]

Where:
    Δlog π = log πθ(yw|x) - log πθ(yl|x) - log πref(yw|x) + log πref(yl|x)

Gradient:
    ∇θ L_DPO = -β E[(σ(Δ̂) - 1)[∇θ log πθ(yw|x) - ∇θ log πθ(yl|x)]]

References:
    - DPO: https://arxiv.org/abs/2305.18290 (Rafailov et al., 2023)
    - Bradley-Terry Model: Bradley & Terry, 1952
    - Related: IPO (https://arxiv.org/abs/2310.12036)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

print("=" * 80)
print("DPO (Direct Preference Optimization) Training")
print("=" * 80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


@dataclass
class DPOConfig:
    """DPO Configuration"""
    # Model settings
    model_name: str = "gpt2"
    ref_model_name: str = "gpt2"  # Usually same as model_name (SFT checkpoint)

    # DPO hyperparameters
    beta: float = 0.1  # Temperature parameter (KL regularization strength)
    learning_rate: float = 5e-7
    max_length: int = 512

    # Training settings
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Logging
    log_every: int = 10


class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs.

    Each sample contains:
        - prompt (x)
        - chosen response (yw) - the preferred response
        - rejected response (yl) - the less preferred response
    """
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Example preference data
        # In practice, use datasets like Anthropic HH-RLHF, OpenAssistant, etc.
        self.data = [
            {
                'prompt': 'Explain what machine learning is in simple terms:',
                'chosen': 'Machine learning is a way for computers to learn from examples and improve at tasks without being explicitly programmed for every situation. The computer finds patterns in data and uses them to make predictions or decisions.',
                'rejected': 'Machine learning is complicated algorithms and math stuff that makes computers smart using neural networks and stuff.'
            },
            {
                'prompt': 'What should I do if I feel stressed?',
                'chosen': 'There are several healthy ways to manage stress: 1) Practice deep breathing or meditation, 2) Exercise regularly, 3) Get enough sleep, 4) Talk to friends or a counselor, 5) Take breaks and do activities you enjoy. Everyone is different, so try different approaches to see what works for you.',
                'rejected': 'Just don\'t think about it. Stress is in your head. You should just ignore it and it will go away.'
            },
            {
                'prompt': 'How do I make chocolate chip cookies?',
                'chosen': 'Here\'s a simple recipe: 1) Mix 1 cup butter, 3/4 cup sugar, 3/4 cup brown sugar. 2) Add 2 eggs and vanilla. 3) Mix in 2 1/4 cups flour, 1 tsp baking soda, 1 tsp salt. 4) Fold in 2 cups chocolate chips. 5) Bake at 375°F for 9-11 minutes. Let cool and enjoy!',
                'rejected': 'Buy cookie dough from the store and follow the instructions on the package. That\'s the easiest way.'
            },
            {
                'prompt': 'Why is the sky blue?',
                'chosen': 'The sky appears blue due to a phenomenon called Rayleigh scattering. Sunlight contains all colors of the rainbow, but blue light has a shorter wavelength. As sunlight passes through the atmosphere, blue light is scattered in all directions more than other colors, making the sky appear blue to our eyes.',
                'rejected': 'The sky is blue because that\'s just the color it is. It reflects the ocean or something like that.'
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt + chosen
        chosen_text = item['prompt'] + " " + item['chosen']
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize prompt + rejected
        rejected_text = item['prompt'] + " " + item['rejected']
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize prompt only (for masking)
        prompt_tokens = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_tokens.input_ids.shape[1]

        return {
            'chosen_input_ids': chosen_tokens.input_ids.squeeze(0),
            'chosen_attention_mask': chosen_tokens.attention_mask.squeeze(0),
            'rejected_input_ids': rejected_tokens.input_ids.squeeze(0),
            'rejected_attention_mask': rejected_tokens.attention_mask.squeeze(0),
            'prompt_length': prompt_length
        }


def compute_log_probs(model, input_ids, attention_mask, prompt_length):
    """
    Compute log probabilities of a sequence.

    We only compute log probs for the response tokens (not the prompt).

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask
        prompt_length: Length of prompt (to mask it out)

    Returns:
        log_probs: Sum of log probabilities for response tokens
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Compute log probabilities
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Get log probs of actual tokens (shifted by 1)
    # Shape: [batch_size, seq_len-1]
    log_probs = torch.gather(
        log_probs_all[:, :-1, :],
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # Create mask for response tokens only
    response_mask = torch.zeros_like(log_probs)
    response_mask[:, prompt_length:] = 1.0

    # Sum log probs only for response tokens
    log_probs = (log_probs * response_mask).sum(dim=-1)

    return log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute DPO loss.

    Mathematical formula:
        L_DPO(θ) = -E[log σ(β Δlog π)]

    Where:
        Δlog π = log πθ(yw|x) - log πθ(yl|x) - log πref(yw|x) + log πref(yl|x)

    Args:
        policy_chosen_logps: Log P(yw|x) from policy
        policy_rejected_logps: Log P(yl|x) from policy
        reference_chosen_logps: Log P(yw|x) from reference
        reference_rejected_logps: Log P(yl|x) from reference
        beta: Temperature parameter

    Returns:
        loss: DPO loss
        stats: Dictionary with statistics
    """
    # Compute log ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO loss: -log σ(β Δlog π)
    # Where Δlog π = policy_logratios - reference_logratios
    logits = beta * (policy_logratios - reference_logratios)

    # Binary cross entropy with logits (numerically stable)
    # -log σ(x) = log(1 + exp(-x))
    loss = -F.logsigmoid(logits).mean()

    # Compute statistics
    with torch.no_grad():
        # Implicit reward (according to DPO reparameterization)
        # r(x, y) = β log(π(y|x) / πref(y|x))
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

        # Accuracy: how often does policy prefer chosen over rejected?
        accuracy = (policy_logratios > 0).float().mean()

    stats = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'chosen_rewards': chosen_rewards.mean().item(),
        'rejected_rewards': rejected_rewards.mean().item(),
        'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
    }

    return loss, stats


def train_epoch(
    model,
    ref_model,
    dataloader,
    optimizer,
    scheduler,
    config: DPOConfig,
    epoch: int
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_stats = {
        'accuracy': 0,
        'chosen_rewards': 0,
        'rejected_rewards': 0,
        'reward_margin': 0
    }

    for step, batch in enumerate(dataloader):
        # Move batch to device
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)
        prompt_length = batch['prompt_length'][0].item()  # Assume same for batch

        # Compute log probs for policy (current model)
        policy_chosen_logps = compute_log_probs(
            model, chosen_input_ids, chosen_attention_mask, prompt_length
        )
        policy_rejected_logps = compute_log_probs(
            model, rejected_input_ids, rejected_attention_mask, prompt_length
        )

        # Compute log probs for reference (frozen model)
        with torch.no_grad():
            reference_chosen_logps = compute_log_probs(
                ref_model, chosen_input_ids, chosen_attention_mask, prompt_length
            )
            reference_rejected_logps = compute_log_probs(
                ref_model, rejected_input_ids, rejected_attention_mask, prompt_length
            )

        # Compute DPO loss
        loss, stats = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=config.beta
        )

        # Backward pass
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        # Update weights
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Track statistics
        total_loss += stats['loss']
        for key in total_stats:
            total_stats[key] += stats[key]

        # Logging
        if (step + 1) % config.log_every == 0:
            avg_loss = total_loss / (step + 1)
            avg_acc = total_stats['accuracy'] / (step + 1)
            print(f"Epoch {epoch+1}, Step {step+1}/{len(dataloader)}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Reward Margin: {total_stats['reward_margin']/(step+1):.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    # Return average statistics
    num_steps = len(dataloader)
    return {
        'loss': total_loss / num_steps,
        'accuracy': total_stats['accuracy'] / num_steps,
        'reward_margin': total_stats['reward_margin'] / num_steps
    }


def main():
    """Main training loop."""
    config = DPOConfig()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Beta (β): {config.beta}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Num Epochs: {config.num_epochs}")

    # Load tokenizer and models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (being trained)
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)

    # Reference model (frozen)
    # In practice, this is usually the SFT model
    ref_model = AutoModelForCausalLM.from_pretrained(config.ref_model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("✓ Models loaded")
    print(f"  Policy model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dataset and dataloader
    dataset = PreferenceDataset(tokenizer, max_length=config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    print(f"\n✓ Dataset loaded: {len(dataset)} preference pairs")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print("\n" + "=" * 80)
    print("DPO Training")
    print("=" * 80)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}")

        stats = train_epoch(model, ref_model, dataloader, optimizer, scheduler, config, epoch)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {stats['loss']:.4f}")
        print(f"  Average Accuracy: {stats['accuracy']:.4f}")
        print(f"  Average Reward Margin: {stats['reward_margin']:.4f}")

        # Sample generation
        print(f"\nSample Generation:")
        test_prompt = "What is the best way to learn programming?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {response}")
        model.train()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Mathematical Summary
    print("\n" + "=" * 80)
    print("Mathematical Summary")
    print("=" * 80)
    print("""
DPO Key Insight:
    Instead of RL, directly optimize the preference probability.

    Given preference: yw ≻ yl (yw preferred over yl)

    DPO Loss:
        L = -E[log σ(β Δlog π)]

    Where:
        Δlog π = log π(yw|x)/πref(yw|x) - log π(yl|x)/πref(yl|x)

    This is equivalent to:
        L = -E[log σ(β[log π(yw|x) - log π(yl|x) - log πref(yw|x) + log πref(yl|x)])]

    Gradient:
        ∇L = -β E[(σ(Δ̂) - 1)[∇log π(yw|x) - ∇log π(yl|x)]]

    Where:
        σ(Δ̂) is the model's predicted preference probability

Advantages over PPO:
    ✓ No reward model needed (preferences used directly)
    ✓ No RL training loop (supervised learning)
    ✓ Simpler implementation
    ✓ More stable training
    ✓ Competitive or better performance

Key Hyperparameter (β):
    - Controls strength of KL penalty
    - Higher β → stay closer to reference model
    - Lower β → optimize preferences more aggressively
    - Typical values: 0.1 to 0.5
    """)

    print("\nNext steps:")
    print("- Use real preference datasets (Anthropic HH-RLHF, OpenAssistant)")
    print("- Try different β values")
    print("- Compare with PPO training")
    print("- Use LoRA for efficient training of large models")
    print("- Experiment with variants: IPO, KTO, etc.")


if __name__ == "__main__":
    main()
