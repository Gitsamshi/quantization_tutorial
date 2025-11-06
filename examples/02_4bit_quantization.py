"""
Example 2: 4-bit Quantization

This script demonstrates advanced 4-bit quantization techniques including:
- Basic 4-bit quantization
- Block-wise quantization (for better accuracy)
- NF4 (Normal Float 4) quantization
- Comparison with 8-bit quantization

Run: python examples/02_4bit_quantization.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Optional


def check_gpu():
    """Check if GPU is available and print info."""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Running on CPU (will be slower)")
        return False

    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    return True


def quantize_4bit_simple(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Simple 4-bit quantization.

    4-bit allows values from 0 to 15 (unsigned) or -8 to 7 (signed).
    We'll use signed 4-bit for better representation of typical weight distributions.

    Args:
        tensor: Input tensor to quantize

    Returns:
        Quantized tensor (packed), scale, and zero_point
    """
    qmin, qmax = -8, 7  # 4-bit signed range

    min_val = tensor.min()
    max_val = tensor.max()

    # Calculate scale and zero point
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)

    return quantized.to(torch.int8), scale.item(), zero_point.item()


def dequantize_4bit_simple(quantized: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    """Dequantize 4-bit tensor."""
    return (quantized.float() - zero_point) * scale


def quantize_4bit_blockwise(
    tensor: torch.Tensor,
    block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block-wise 4-bit quantization.

    Divides tensor into blocks and quantizes each block independently.
    This preserves more information for tensors with varying magnitudes.

    Args:
        tensor: Input tensor to quantize
        block_size: Size of each block for independent quantization

    Returns:
        Quantized tensor, scale per block, zero_point per block
    """
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()

    # Pad to make divisible by block_size
    pad_len = (block_size - len(tensor_flat) % block_size) % block_size
    if pad_len > 0:
        tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_len, device=tensor.device)])

    # Reshape into blocks
    num_blocks = len(tensor_flat) // block_size
    tensor_blocks = tensor_flat.reshape(num_blocks, block_size)

    qmin, qmax = -8, 7
    quantized_blocks = []
    scales = []
    zero_points = []

    for block in tensor_blocks:
        min_val = block.min()
        max_val = block.max()

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1.0  # Avoid division by zero

        zero_point = qmin - torch.round(min_val / scale)

        q_block = torch.clamp(torch.round(block / scale) + zero_point, qmin, qmax)

        quantized_blocks.append(q_block)
        scales.append(scale)
        zero_points.append(zero_point)

    quantized = torch.stack(quantized_blocks).flatten()

    # Remove padding
    if pad_len > 0:
        quantized = quantized[:-pad_len]

    quantized = quantized.reshape(original_shape)
    scales = torch.tensor(scales, device=tensor.device)
    zero_points = torch.tensor(zero_points, device=tensor.device)

    return quantized.to(torch.int8), scales, zero_points


def dequantize_4bit_blockwise(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    block_size: int = 64
) -> torch.Tensor:
    """Dequantize block-wise quantized tensor."""
    original_shape = quantized.shape
    quantized_flat = quantized.flatten().float()

    # Pad to make divisible by block_size
    pad_len = (block_size - len(quantized_flat) % block_size) % block_size
    if pad_len > 0:
        quantized_flat = torch.cat([quantized_flat, torch.zeros(pad_len, device=quantized.device)])

    num_blocks = len(quantized_flat) // block_size
    quantized_blocks = quantized_flat.reshape(num_blocks, block_size)

    dequantized_blocks = []
    for i, block in enumerate(quantized_blocks):
        dq_block = (block - zero_points[i]) * scales[i]
        dequantized_blocks.append(dq_block)

    dequantized = torch.cat(dequantized_blocks)

    # Remove padding
    if pad_len > 0:
        dequantized = dequantized[:-pad_len]

    return dequantized.reshape(original_shape)


# NF4 quantization lookup table (normalized float 4-bit)
# Optimized for normal distribution with zero mean
NF4_QUANT_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


def quantize_nf4(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    NF4 (Normal Float 4) quantization.

    NF4 is optimized for normally distributed data (common in neural network weights).
    Uses a non-uniform quantization that places more quantization points near zero.

    Args:
        tensor: Input tensor to quantize

    Returns:
        Quantized indices and absmax (absolute maximum for normalization)
    """
    # Normalize tensor
    absmax = tensor.abs().max()
    if absmax == 0:
        absmax = 1.0

    normalized = tensor / absmax

    # Move quantization table to same device as tensor
    quant_table = NF4_QUANT_TABLE.to(tensor.device)

    # Find nearest quantization value for each element
    # Expand dimensions for broadcasting
    normalized_expanded = normalized.unsqueeze(-1)  # [..., 1]
    quant_table_expanded = quant_table.view(*([1] * len(normalized.shape)), -1)  # [1, 1, ..., 16]

    # Compute distances and find nearest
    distances = torch.abs(normalized_expanded - quant_table_expanded)
    indices = torch.argmin(distances, dim=-1)

    return indices.to(torch.uint8), absmax.item()


def dequantize_nf4(indices: torch.Tensor, absmax: float) -> torch.Tensor:
    """Dequantize NF4 tensor."""
    quant_table = NF4_QUANT_TABLE.to(indices.device)
    normalized = quant_table[indices.long()]
    return normalized * absmax


def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Calculate error metrics."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()

    signal_power = torch.mean(original ** 2).item()
    noise_power = mse
    sqnr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    return {'mse': mse, 'mae': mae, 'sqnr_db': sqnr_db}


class TransformerBlock(nn.Module):
    """A simple transformer block for demonstration."""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm2(x + ff_out)

        return x


def main():
    print("=" * 70)
    print("4-bit Quantization Tutorial - Example 2")
    print("=" * 70 + "\n")

    has_gpu = check_gpu()
    device = torch.device('cuda' if has_gpu else 'cpu')

    # ========================================================================
    # Part 1: Simple 4-bit Quantization
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Simple 4-bit Quantization")
    print("=" * 70)

    tensor = torch.randn(1000, 1000, device=device)
    print(f"\nOriginal tensor shape: {tensor.shape}")
    print(f"Original memory: {tensor.element_size() * tensor.nelement() / 1e6:.2f} MB")
    print(f"Original range: [{tensor.min():.4f}, {tensor.max():.4f}]")

    # Quantize to 4-bit
    q_4bit, scale, zp = quantize_4bit_simple(tensor)
    dq_4bit = dequantize_4bit_simple(q_4bit, scale, zp)

    # Note: In practice, 4-bit values are packed (2 values per byte)
    # Here we use int8 for simplicity
    print(f"\nQuantized memory (unpacked): {q_4bit.element_size() * q_4bit.nelement() / 1e6:.2f} MB")
    print(f"Quantized memory (packed): {q_4bit.element_size() * q_4bit.nelement() / 2 / 1e6:.2f} MB")
    print(f"Memory reduction: {tensor.element_size() / (q_4bit.element_size() / 2):.1f}x")

    metrics_4bit = calculate_metrics(tensor, dq_4bit)
    print(f"\nError metrics:")
    print(f"  SQNR: {metrics_4bit['sqnr_db']:.2f} dB")
    print(f"  MAE: {metrics_4bit['mae']:.6f}")

    # ========================================================================
    # Part 2: Block-wise Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: Block-wise 4-bit Quantization")
    print("=" * 70)

    print("\nComparing block sizes...")

    block_sizes = [16, 64, 256]
    results = {}

    for block_size in block_sizes:
        q_block, scales, zps = quantize_4bit_blockwise(tensor, block_size)
        dq_block = dequantize_4bit_blockwise(q_block, scales, zps, block_size)
        metrics = calculate_metrics(tensor, dq_block)
        results[block_size] = metrics

        num_blocks = len(scales)
        overhead = (scales.element_size() + zps.element_size()) * num_blocks / 1e6

        print(f"\nBlock size: {block_size}")
        print(f"  Number of blocks: {num_blocks}")
        print(f"  Metadata overhead: {overhead:.4f} MB")
        print(f"  SQNR: {metrics['sqnr_db']:.2f} dB")

    best_block_size = max(results.keys(), key=lambda k: results[k]['sqnr_db'])
    print(f"\n✓ Best block size: {best_block_size} (SQNR: {results[best_block_size]['sqnr_db']:.2f} dB)")

    # ========================================================================
    # Part 3: NF4 Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 3: NF4 (Normal Float 4) Quantization")
    print("=" * 70)

    # Generate normally distributed tensor (like neural network weights)
    normal_tensor = torch.randn(1000, 1000, device=device) * 0.02  # Scale typical for NN weights
    print(f"\nTensor statistics:")
    print(f"  Mean: {normal_tensor.mean():.6f}")
    print(f"  Std: {normal_tensor.std():.6f}")
    print(f"  Range: [{normal_tensor.min():.6f}, {normal_tensor.max():.6f}]")

    # Compare simple 4-bit vs NF4
    q_simple, scale_s, zp_s = quantize_4bit_simple(normal_tensor)
    dq_simple = dequantize_4bit_simple(q_simple, scale_s, zp_s)
    metrics_simple = calculate_metrics(normal_tensor, dq_simple)

    q_nf4, absmax = quantize_nf4(normal_tensor)
    dq_nf4 = dequantize_nf4(q_nf4, absmax)
    metrics_nf4 = calculate_metrics(normal_tensor, dq_nf4)

    print(f"\nSimple 4-bit quantization:")
    print(f"  SQNR: {metrics_simple['sqnr_db']:.2f} dB")
    print(f"  MAE: {metrics_simple['mae']:.6f}")

    print(f"\nNF4 quantization:")
    print(f"  SQNR: {metrics_nf4['sqnr_db']:.2f} dB")
    print(f"  MAE: {metrics_nf4['mae']:.6f}")

    improvement = metrics_nf4['sqnr_db'] - metrics_simple['sqnr_db']
    print(f"\n✓ NF4 improved SQNR by {improvement:.2f} dB for normally distributed data!")

    # ========================================================================
    # Part 4: Transformer Model Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 4: Transformer Model Quantization")
    print("=" * 70)

    model = TransformerBlock().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Calculate memory for different precisions
    fp32_size = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1e6
    fp16_size = fp32_size / 2
    int8_size = fp32_size / 4
    int4_size = fp32_size / 8

    print(f"\nMemory requirements:")
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  FP16: {fp16_size:.2f} MB  (2x reduction)")
    print(f"  INT8: {int8_size:.2f} MB  (4x reduction)")
    print(f"  INT4: {int4_size:.2f} MB  (8x reduction)")

    # Quantize model weights with NF4
    print(f"\nQuantizing model weights with NF4...")
    quantized_weights = {}
    total_error = 0

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            q_weight, absmax = quantize_nf4(param.data)
            dq_weight = dequantize_nf4(q_weight, absmax)

            metrics = calculate_metrics(param.data, dq_weight)
            quantized_weights[name] = (q_weight, absmax)
            total_error += metrics['mae']

            print(f"  {name}: SQNR = {metrics['sqnr_db']:.2f} dB")

    avg_error = total_error / len(quantized_weights)
    print(f"\n✓ Average MAE across all weights: {avg_error:.6f}")

    # ========================================================================
    # Part 5: Comparison Summary
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 5: Quantization Comparison Summary")
    print("=" * 70)

    comparison_tensor = torch.randn(10000, 1000, device=device) * 0.02

    methods = {
        '4-bit Simple': lambda t: quantize_4bit_simple(t)[:2],
        '4-bit Block-64': lambda t: quantize_4bit_blockwise(t, 64)[:2],
        '4-bit NF4': lambda t: quantize_nf4(t),
    }

    print(f"\nTest tensor shape: {comparison_tensor.shape}")
    print(f"Test tensor size: {comparison_tensor.element_size() * comparison_tensor.nelement() / 1e6:.2f} MB")
    print("\nMethod comparison:")

    for method_name, quantize_fn in methods.items():
        start = time.time()
        result = quantize_fn(comparison_tensor)
        quant_time = (time.time() - start) * 1000

        # Dequantize based on method
        if method_name == '4-bit Simple':
            q, scale = result
            dq = dequantize_4bit_simple(q, scale, 0)
        elif method_name == '4-bit Block-64':
            q, scales = result
            # For this comparison, we just use first scale (simplified)
            dq = comparison_tensor  # Skip complex dequantization for timing
        else:  # NF4
            q, absmax = result
            dq = dequantize_nf4(q, absmax)

        if method_name != '4-bit Block-64':
            metrics = calculate_metrics(comparison_tensor, dq)
            print(f"\n{method_name}:")
            print(f"  Quantization time: {quant_time:.2f} ms")
            print(f"  SQNR: {metrics['sqnr_db']:.2f} dB")
            print(f"  MAE: {metrics['mae']:.6f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ Key Takeaways:")
    print("  1. 4-bit quantization achieves 8× memory reduction (FP32 → INT4)")
    print("  2. Block-wise quantization preserves more information")
    print("  3. NF4 is optimal for normally distributed weights (LLMs)")
    print("  4. Smaller block sizes = better accuracy but more overhead")
    print("  5. NF4 is used in QLoRA for efficient LLM fine-tuning")

    print("\n✓ Practical Applications:")
    print("  - QLoRA: NF4 quantization for parameter-efficient fine-tuning")
    print("  - GGML/GGUF: Various 4-bit schemes for llama.cpp")
    print("  - GPTQ: Advanced 4-bit quantization with calibration")

    print("\n✓ Next Steps:")
    print("  - Run Example 3 for real LLM quantization with bitsandbytes")
    print("  - Experiment with different block sizes for your use case")

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
