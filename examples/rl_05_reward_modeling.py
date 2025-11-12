"""
Example 5: Reward Model Training

This example demonstrates how to train a reward model from human preference data.
The reward model is a crucial component in RLHF pipelines.

Mathematical Background:
========================

Bradley-Terry Preference Model:
    The probability that response yw is preferred over yl is:

    P(yw ≻ yl | x) = σ(r(x, yw) - r(x, yl))

    Where:
        - σ: Sigmoid function
        - r(x, y): Reward model's score for response y given prompt x

Reward Model Loss (Binary Cross-Entropy):
    L(θ) = -E[log σ(r_θ(x, yw) - r_θ(x, yl))]

    Equivalently:
    L(θ) = -E[log P(yw ≻ yl | x)]

Gradient:
    The gradient pushes the model to assign higher scores to preferred responses:

    ∇θ L = -E[(1 - σ(Δr)) · (∇θ r(x, yw) - ∇θ r(x, yl))]

    Where Δr = r(x, yw) - r(x, yl)

Training Process:
    1. Collect preference data: (x, yw, yl) where yw ≻ yl
    2. Initialize reward model r_θ (typically from SFT model + value head)
    3. Optimize to maximize P(yw ≻ yl) over dataset
    4. Use trained r_θ in RL phase (PPO, etc.)

References:
    - InstructGPT: https://arxiv.org/abs/2203.02155 (Ouyang et al., 2022)
    - Learning to Summarize: https://arxiv.org/abs/2009.01325 (Stiennon et al., 2020)
    - Scaling Laws for RM: https://arxiv.org/abs/2210.10760 (Gao et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

print("=" * 80)
print("Reward Model Training")
print("=" * 80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


@dataclass
class RewardModelConfig:
    """Reward Model Configuration"""
    # Model settings
    base_model: str = "gpt2"

    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Logging
    log_every: int = 10


class RewardModel(nn.Module):
    """
    Reward Model: Transformer + Scalar Head

    Architecture:
        Input: [x, y] (prompt + response)
        → Transformer encoder
        → Last token representation
        → Linear layer
        → Output: scalar reward r(x, y)
    """
    def __init__(self, base_model_name: str):
        super().__init__()

        # Load base model (e.g., GPT-2, LLaMA)
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # Reward head: map hidden states to scalar
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

        # Initialize reward head with small weights
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass to compute reward.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            rewards: Scalar reward for each sequence [batch_size]
        """
        # Get transformer outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Use representation of last token
        # (For causal models, this summarizes the whole sequence)
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
        else:
            # Use last token
            last_hidden = hidden_states[:, -1, :]

        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards


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
        # Format: (prompt, chosen, rejected)
        # In practice, use datasets like Anthropic HH-RLHF, OpenAssistant
        self.data = [
            {
                'prompt': 'What is the best way to learn programming?',
                'chosen': 'The best way to learn programming is through hands-on practice. Start with fundamentals like variables and control flow, then build small projects. Code every day, read others\' code, and don\'t be afraid to make mistakes. Online resources like documentation, tutorials, and coding challenges are invaluable.',
                'rejected': 'Just watch YouTube videos and you\'ll become a programmer. You don\'t really need to practice much.'
            },
            {
                'prompt': 'How can I reduce stress?',
                'chosen': 'There are many evidence-based ways to reduce stress: 1) Regular exercise releases endorphins, 2) Practice mindfulness or meditation, 3) Maintain good sleep hygiene, 4) Connect with friends and family, 5) Consider therapy if needed. Different techniques work for different people, so experiment to find what helps you.',
                'rejected': 'Stress isn\'t real, it\'s all in your head. Just ignore it and it will disappear.'
            },
            {
                'prompt': 'Explain photosynthesis.',
                'chosen': 'Photosynthesis is the process by which plants convert light energy into chemical energy. In the chloroplasts, plants use sunlight, water (H₂O), and carbon dioxide (CO₂) to produce glucose (C₆H₁₂O₆) and oxygen (O₂). The light-dependent reactions occur in the thylakoid membranes, while the Calvin cycle happens in the stroma.',
                'rejected': 'Photosynthesis is when plants eat sunlight to grow. They breathe in oxygen and breathe out CO2, opposite of animals.'
            },
            {
                'prompt': 'What should I do if my computer won\'t start?',
                'chosen': 'First, check the basics: 1) Ensure it\'s plugged in and the outlet works, 2) Try holding the power button for 10 seconds to reset, 3) Check for loose connections, 4) Listen for beep codes or LED indicators. If it still won\'t start, the issue could be the power supply, motherboard, or RAM. Consider seeking professional help if you\'re not comfortable troubleshooting hardware.',
                'rejected': 'Hit it a few times. If that doesn\'t work, throw it away and buy a new one.'
            },
            {
                'prompt': 'How do I make coffee?',
                'chosen': 'Here\'s a simple method for drip coffee: 1) Use fresh, cold water and quality coffee beans, 2) Grind beans to medium coarseness just before brewing, 3) Use about 1-2 tablespoons of coffee per 6 oz of water, 4) Place filter in dripper and add grounds, 5) Pour hot water (195-205°F) over grounds, 6) Let brew for 4-5 minutes. Adjust ratios to taste.',
                'rejected': 'Just put some instant coffee in a cup and add hot water. That\'s all you need to know.'
            },
            {
                'prompt': 'Why is the sky blue?',
                'chosen': 'The sky appears blue due to Rayleigh scattering. Sunlight is made of all colors of the spectrum. As it passes through the atmosphere, shorter wavelengths (blue and violet) scatter more than longer wavelengths (red and orange). We see blue rather than violet because our eyes are more sensitive to blue, and some violet light is absorbed by the upper atmosphere.',
                'rejected': 'The sky is blue because it reflects the ocean. That\'s why when you\'re far from the ocean, the sky looks less blue.'
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

        return {
            'chosen_input_ids': chosen_tokens.input_ids.squeeze(0),
            'chosen_attention_mask': chosen_tokens.attention_mask.squeeze(0),
            'rejected_input_ids': rejected_tokens.input_ids.squeeze(0),
            'rejected_attention_mask': rejected_tokens.attention_mask.squeeze(0),
        }


def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute reward model loss using Bradley-Terry model.

    Mathematical formula:
        L = -log σ(r(x, yw) - r(x, yl))
        L = -log P(yw ≻ yl)

    This is equivalent to binary cross-entropy.

    Args:
        chosen_rewards: Rewards for chosen responses r(x, yw)
        rejected_rewards: Rewards for rejected responses r(x, yl)

    Returns:
        loss: Reward model loss
        stats: Dictionary with statistics
    """
    # Difference in rewards
    logits = chosen_rewards - rejected_rewards

    # Loss: -log σ(logits)
    # Using logsigmoid for numerical stability
    loss = -F.logsigmoid(logits).mean()

    # Compute statistics
    with torch.no_grad():
        # Accuracy: how often is chosen reward > rejected reward?
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        # Margin: average difference
        margin = (chosen_rewards - rejected_rewards).mean()

    stats = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'margin': margin.item(),
        'chosen_reward_mean': chosen_rewards.mean().item(),
        'rejected_reward_mean': rejected_rewards.mean().item()
    }

    return loss, stats


def train_epoch(
    model: RewardModel,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    config: RewardModelConfig,
    epoch: int
):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_stats = {
        'accuracy': 0,
        'margin': 0,
        'chosen_reward_mean': 0,
        'rejected_reward_mean': 0
    }

    for step, batch in enumerate(dataloader):
        # Move batch to device
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)

        # Compute rewards for chosen responses
        chosen_rewards = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )

        # Compute rewards for rejected responses
        rejected_rewards = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )

        # Compute loss
        loss, stats = reward_model_loss(chosen_rewards, rejected_rewards)

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
            avg_margin = total_stats['margin'] / (step + 1)

            print(f"Epoch {epoch+1}, Step {step+1}/{len(dataloader)}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Margin: {avg_margin:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    # Return average statistics
    num_steps = len(dataloader)
    return {
        'loss': total_loss / num_steps,
        'accuracy': total_stats['accuracy'] / num_steps,
        'margin': total_stats['margin'] / num_steps
    }


def evaluate_model(model: RewardModel, tokenizer, test_samples: List[Dict]):
    """
    Evaluate reward model on test samples.

    Args:
        model: Trained reward model
        tokenizer: Tokenizer
        test_samples: List of (prompt, response1, response2) with ground truth preference
    """
    model.eval()

    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    correct = 0
    total = 0

    for sample in test_samples:
        prompt = sample['prompt']
        response1 = sample['response1']
        response2 = sample['response2']
        preferred = sample['preferred']  # 1 or 2

        # Tokenize
        text1 = prompt + " " + response1
        text2 = prompt + " " + response2

        tokens1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        tokens2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        # Compute rewards
        with torch.no_grad():
            reward1 = model(**tokens1).item()
            reward2 = model(**tokens2).item()

        # Check if prediction matches preference
        predicted_preferred = 1 if reward1 > reward2 else 2

        if predicted_preferred == preferred:
            correct += 1
        total += 1

        print(f"\nPrompt: {prompt}")
        print(f"Response 1: {response1[:100]}... | Reward: {reward1:.4f}")
        print(f"Response 2: {response2[:100]}... | Reward: {reward2:.4f}")
        print(f"Preferred: Response {preferred} | Predicted: Response {predicted_preferred}")
        print(f"✓ Correct" if predicted_preferred == preferred else "✗ Wrong")

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.2%}")


def main():
    """Main training loop."""
    config = RewardModelConfig()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Base Model: {config.base_model}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Num Epochs: {config.num_epochs}")

    # Load tokenizer
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Create reward model
    model = RewardModel(config.base_model).to(device)

    print("✓ Model loaded")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
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
    print("Reward Model Training")
    print("=" * 80)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}")

        stats = train_epoch(model, dataloader, optimizer, scheduler, config, epoch)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {stats['loss']:.4f}")
        print(f"  Average Accuracy: {stats['accuracy']:.4f}")
        print(f"  Average Margin: {stats['margin']:.4f}")

    # Evaluation on test samples
    test_samples = [
        {
            'prompt': 'What is the best programming language?',
            'response1': 'The "best" programming language depends on your goals. Python is great for beginners and data science, JavaScript for web development, C++ for performance-critical applications, and Rust for systems programming. Choose based on your project needs.',
            'response2': 'Python is objectively the best language ever created. All other languages are terrible and you should never use them.',
            'preferred': 1
        },
        {
            'prompt': 'How do I stay motivated?',
            'response1': 'Set clear, achievable goals and break them into smaller tasks. Celebrate small wins, maintain a routine, surround yourself with supportive people, and remember your "why". It\'s normal to have ups and downs - be kind to yourself.',
            'response2': 'Motivation is overrated. Just force yourself to work even when you don\'t want to. That\'s what successful people do.',
            'preferred': 1
        }
    ]

    evaluate_model(model, tokenizer, test_samples)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Mathematical Summary
    print("\n" + "=" * 80)
    print("Mathematical Summary")
    print("=" * 80)
    print("""
Reward Model Objective:
    Learn to predict human preferences using the Bradley-Terry model.

Bradley-Terry Model:
    P(yw ≻ yl | x) = σ(r(x, yw) - r(x, yl))

    Where:
        - yw: Preferred (chosen) response
        - yl: Less preferred (rejected) response
        - r(x, y): Reward model score
        - σ: Sigmoid function

Loss Function:
    L(θ) = -E[log σ(r_θ(x, yw) - r_θ(x, yl))]

    Equivalent to:
    L(θ) = -E[log P(yw ≻ yl | x)]

Gradient:
    ∇θ L = -E[(1 - σ(Δr)) · (∇θ r(yw) - ∇θ r(yl))]

    Intuition:
        - If Δr > 0 (correct ranking): small gradient
        - If Δr < 0 (wrong ranking): large gradient pushing correction

Architecture:
    Input → Transformer → Last Token Embedding → Linear Head → Scalar Reward

Training Process:
    1. Collect human preferences: (x, yw, yl)
    2. Train reward model to rank yw > yl
    3. Use trained reward model in RL phase

Key Insights:
    ✓ Reward model learns implicit human preferences
    ✓ Can generalize to new prompts (not in training data)
    ✓ Quality depends on preference data quality
    ✓ Can suffer from overoptimization (Goodhart's law)

Common Issues:
    - Distribution shift: RM trained on SFT, used on RL policy
    - Overoptimization: RL exploits RM weaknesses
    - Reward hacking: Policy finds shortcuts

Solutions:
    → Use KL penalty to stay close to SFT model
    → Iterative RLHF (retrain RM on new policy outputs)
    → Ensemble reward models
    → DPO (bypass RM entirely)
    """)

    print("\nNext steps:")
    print("- Use real preference datasets (Anthropic HH-RLHF, etc.)")
    print("- Train larger reward models")
    print("- Use reward model in PPO training (see rl_02_ppo_training.py)")
    print("- Implement ensemble reward models for robustness")
    print("- Study reward model overoptimization")


if __name__ == "__main__":
    main()
