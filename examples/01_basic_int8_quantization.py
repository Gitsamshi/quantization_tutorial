"""
Example 1: Basic INT8 Quantization

This script demonstrates the fundamentals of INT8 quantization including:
- Symmetric quantization
- Asymmetric quantization
- Quantization error analysis
- GPU-accelerated quantization

Run: python examples/01_basic_int8_quantization.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple


def check_gpu():
    """Check if GPU is available and print info."""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Running on CPU (will be slower)")
        return False

    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    return True


def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float]:
    """
    Symmetric quantization: maps values symmetrically around zero.

    Formula: Q(x) = round(x / S)
    where S = max(|x|) / (2^(bits-1) - 1)

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization (default: 8)

    Returns:
        Quantized tensor and scale factor
    """
    # Calculate scale factor
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)

    max_val = torch.max(torch.abs(tensor))
    scale = max_val / qmax

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)

    return quantized.to(torch.int8), scale.item()


def symmetric_dequantize(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Dequantize a symmetrically quantized tensor.

    Formula: x̂ = S * Q(x)
    """
    return quantized.float() * scale


def asymmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, int]:
    """
    Asymmetric quantization: maps values with separate min/max.
    Better for tensors with asymmetric distributions.

    Formula: Q(x) = round(x / S) - Z
    where:
        S = (max - min) / (2^bits - 1)
        Z = -round(min / S)

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization

    Returns:
        Quantized tensor, scale factor, and zero point
    """
    qmax = 2 ** bits - 1
    qmin = 0

    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Calculate scale and zero point
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)

    return quantized.to(torch.uint8), scale.item(), int(zero_point.item())


def asymmetric_dequantize(quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """
    Dequantize an asymmetrically quantized tensor.

    Formula: x̂ = S * (Q(x) - Z)
    """
    return (quantized.float() - zero_point) * scale


def calculate_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Calculate quantization error metrics."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    max_error = torch.max(torch.abs(original - reconstructed)).item()

    # Signal-to-Quantization-Noise Ratio (SQNR)
    signal_power = torch.mean(original ** 2).item()
    noise_power = mse
    sqnr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'sqnr_db': sqnr_db
    }


class SimpleNN(nn.Module):
    """A simple neural network for demonstration."""
    def __init__(self, input_size=784, hidden_size=512, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantize_model_weights(model: nn.Module, method: str = 'symmetric') -> dict:
    """
    Quantize all weights in a model.

    Args:
        model: PyTorch model to quantize
        method: 'symmetric' or 'asymmetric'

    Returns:
        Dictionary containing quantized weights and metadata
    """
    quantized_state = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'symmetric':
                q_weight, scale = symmetric_quantize(param.data)
                quantized_state[name] = {
                    'quantized': q_weight,
                    'scale': scale,
                    'method': 'symmetric'
                }
            elif method == 'asymmetric':
                q_weight, scale, zero_point = asymmetric_quantize(param.data)
                quantized_state[name] = {
                    'quantized': q_weight,
                    'scale': scale,
                    'zero_point': zero_point,
                    'method': 'asymmetric'
                }
        else:
            # Keep biases in float
            quantized_state[name] = {'float': param.data}

    return quantized_state


def main():
    print("=" * 70)
    print("INT8 Quantization Tutorial - Example 1")
    print("=" * 70 + "\n")

    # Check GPU
    has_gpu = check_gpu()
    device = torch.device('cuda' if has_gpu else 'cpu')

    # ========================================================================
    # Part 1: Basic Tensor Quantization
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Basic Tensor Quantization")
    print("=" * 70)

    # Create a sample tensor with various distributions
    print("\n1.1 Symmetric Distribution (Normal)")
    symmetric_tensor = torch.randn(1000, 1000, device=device)
    print(f"Original tensor shape: {symmetric_tensor.shape}")
    print(f"Original tensor range: [{symmetric_tensor.min():.4f}, {symmetric_tensor.max():.4f}]")
    print(f"Original memory: {symmetric_tensor.element_size() * symmetric_tensor.nelement() / 1e6:.2f} MB")

    # Symmetric quantization
    q_sym, scale_sym = symmetric_quantize(symmetric_tensor)
    dq_sym = symmetric_dequantize(q_sym, scale_sym)

    print(f"\nAfter symmetric quantization:")
    print(f"Quantized memory: {q_sym.element_size() * q_sym.nelement() / 1e6:.2f} MB")
    print(f"Memory reduction: {symmetric_tensor.element_size() / q_sym.element_size():.1f}x")
    print(f"Scale factor: {scale_sym:.6f}")

    metrics_sym = calculate_error_metrics(symmetric_tensor, dq_sym)
    print(f"\nError metrics:")
    print(f"  MSE: {metrics_sym['mse']:.6f}")
    print(f"  MAE: {metrics_sym['mae']:.6f}")
    print(f"  Max Error: {metrics_sym['max_error']:.6f}")
    print(f"  SQNR: {metrics_sym['sqnr_db']:.2f} dB")

    # Asymmetric distribution
    print("\n" + "-" * 70)
    print("1.2 Asymmetric Distribution (ReLU-like)")
    asymmetric_tensor = torch.relu(torch.randn(1000, 1000, device=device))
    print(f"Original tensor range: [{asymmetric_tensor.min():.4f}, {asymmetric_tensor.max():.4f}]")

    # Compare symmetric vs asymmetric quantization
    q_sym2, scale_sym2 = symmetric_quantize(asymmetric_tensor)
    dq_sym2 = symmetric_dequantize(q_sym2, scale_sym2)

    q_asym, scale_asym, zp_asym = asymmetric_quantize(asymmetric_tensor)
    dq_asym = asymmetric_dequantize(q_asym, scale_asym, zp_asym)

    print(f"\nSymmetric quantization:")
    metrics_sym2 = calculate_error_metrics(asymmetric_tensor, dq_sym2)
    print(f"  SQNR: {metrics_sym2['sqnr_db']:.2f} dB")
    print(f"  MAE: {metrics_sym2['mae']:.6f}")

    print(f"\nAsymmetric quantization:")
    metrics_asym = calculate_error_metrics(asymmetric_tensor, dq_asym)
    print(f"  SQNR: {metrics_asym['sqnr_db']:.2f} dB")
    print(f"  MAE: {metrics_asym['mae']:.6f}")
    print(f"  Scale: {scale_asym:.6f}, Zero point: {zp_asym}")

    improvement = metrics_asym['sqnr_db'] - metrics_sym2['sqnr_db']
    print(f"\n✓ Asymmetric quantization improved SQNR by {improvement:.2f} dB for asymmetric data!")

    # ========================================================================
    # Part 2: Neural Network Weight Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: Neural Network Weight Quantization")
    print("=" * 70)

    # Create and initialize model
    model = SimpleNN().to(device)
    print(f"\nCreated model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Calculate original size
    original_size = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1e6
    print(f"Original model size: {original_size:.2f} MB")

    # Quantize weights
    print("\nQuantizing model weights (symmetric)...")
    quantized_state = quantize_model_weights(model, method='symmetric')

    # Calculate quantized size
    quantized_size = sum(
        v['quantized'].element_size() * v['quantized'].nelement()
        for v in quantized_state.values()
        if 'quantized' in v
    ) / 1e6
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {original_size / quantized_size:.2f}x")

    # ========================================================================
    # Part 3: Inference Speed Comparison
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 3: Inference Speed Comparison")
    print("=" * 70)

    # Prepare test data
    batch_size = 128
    test_input = torch.randn(batch_size, 784, device=device)

    # Warmup
    for _ in range(10):
        _ = model(test_input)

    if has_gpu:
        torch.cuda.synchronize()

    # Benchmark FP32
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(test_input)
    if has_gpu:
        torch.cuda.synchronize()
    fp32_time = (time.time() - start) / num_iterations

    print(f"\nFP32 inference time: {fp32_time * 1000:.2f} ms")
    print(f"FP32 throughput: {batch_size / fp32_time:.0f} samples/sec")

    # Note: For true INT8 speedup, we would need to use quantized ops
    # This is demonstrated in later examples with bitsandbytes
    print("\nℹ️  Note: For actual INT8 inference speedup, use specialized libraries")
    print("   (demonstrated in Example 3 with bitsandbytes)")

    # ========================================================================
    # Part 4: Per-Channel vs Per-Tensor Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 4: Per-Channel vs Per-Tensor Quantization")
    print("=" * 70)

    # Get first layer weights
    weight = model.fc1.weight.data
    print(f"\nLayer weight shape: {weight.shape}")

    # Per-tensor quantization
    q_tensor, scale_tensor = symmetric_quantize(weight)
    dq_tensor = symmetric_dequantize(q_tensor, scale_tensor)
    metrics_tensor = calculate_error_metrics(weight, dq_tensor)

    print(f"\nPer-tensor quantization:")
    print(f"  Single scale factor: {scale_tensor:.6f}")
    print(f"  SQNR: {metrics_tensor['sqnr_db']:.2f} dB")

    # Per-channel quantization
    print(f"\nPer-channel quantization:")
    per_channel_quantized = []
    scales = []

    for i in range(weight.shape[0]):
        channel = weight[i:i+1, :]
        q_channel, scale = symmetric_quantize(channel)
        per_channel_quantized.append(q_channel)
        scales.append(scale)

    # Dequantize per-channel
    dq_channels = []
    for q_channel, scale in zip(per_channel_quantized, scales):
        dq_channels.append(symmetric_dequantize(q_channel, scale))

    dq_per_channel = torch.cat(dq_channels, dim=0)
    metrics_channel = calculate_error_metrics(weight, dq_per_channel)

    print(f"  {len(scales)} scale factors (one per output channel)")
    print(f"  Scale range: [{min(scales):.6f}, {max(scales):.6f}]")
    print(f"  SQNR: {metrics_channel['sqnr_db']:.2f} dB")

    improvement = metrics_channel['sqnr_db'] - metrics_tensor['sqnr_db']
    print(f"\n✓ Per-channel quantization improved SQNR by {improvement:.2f} dB!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ Key Takeaways:")
    print("  1. INT8 quantization reduces memory by 4x (FP32 → INT8)")
    print("  2. Asymmetric quantization better for non-symmetric distributions")
    print("  3. Per-channel quantization preserves more information")
    print("  4. Quantization introduces small errors (trade-off for efficiency)")
    print("  5. GPU acceleration works for both quantized and non-quantized ops")

    print("\n✓ Next Steps:")
    print("  - Run Example 2 for 4-bit quantization")
    print("  - Run Example 3 for real LLM quantization with bitsandbytes")

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
