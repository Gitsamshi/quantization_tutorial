"""
Example 4: GPTQ Quantization

GPTQ (Generative Pre-trained Transformer Quantization) is an advanced
post-training quantization method that uses calibration data to minimize
quantization error.

This script demonstrates:
- How GPTQ differs from simple quantization
- Using the auto-gptq library
- Comparing GPTQ with other quantization methods

Run: python examples/04_gptq_quantization.py

Note: GPTQ quantization requires calibration and can take several minutes.
"""

import torch
import time
import os
from typing import Optional


def check_dependencies():
    """Check if required libraries are installed."""
    required = {
        'transformers': 'transformers',
        'auto_gptq': 'auto-gptq',
        'accelerate': 'accelerate'
    }

    all_available = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"❌ {module} not available. Install with: pip install {package}")
            all_available = False

    return all_available


def check_gpu():
    """Check GPU availability."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPTQ requires a GPU.")
        return False

    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    return True


def get_gpu_memory():
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_gpu_memory():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def demonstrate_gptq_concept():
    """
    Demonstrate the key concept behind GPTQ using a simple example.
    GPTQ minimizes the squared error: ||WX - W_q X||^2
    where W is original weight and W_q is quantized weight.
    """
    print("\n" + "=" * 70)
    print("GPTQ CONCEPT: Layer-wise Quantization with Error Compensation")
    print("=" * 70)

    # Create a simple weight matrix
    W = torch.randn(512, 512, device='cuda')
    X = torch.randn(512, 1000, device='cuda')  # Calibration data

    print(f"\nWeight matrix: {W.shape}")
    print(f"Calibration data: {X.shape}")

    # Original output
    Y_original = W @ X

    # Simple quantization (naive approach)
    print("\n1. Naive Quantization:")
    scale_naive = W.abs().max() / 127
    W_naive = torch.clamp(torch.round(W / scale_naive), -128, 127)
    W_naive_dq = W_naive * scale_naive
    Y_naive = W_naive_dq @ X

    error_naive = torch.mean((Y_original - Y_naive) ** 2).item()
    print(f"   MSE: {error_naive:.6f}")

    # GPTQ-style quantization (simplified)
    # Process columns sequentially and compensate errors
    print("\n2. GPTQ-style Quantization (Simplified):")

    W_gptq = W.clone()
    W_gptq_quantized = torch.zeros_like(W)

    # Compute Hessian (second-order information from calibration data)
    H = 2 * X @ X.T / X.shape[1]
    H_inv = torch.inverse(H + 1e-4 * torch.eye(H.shape[0], device=H.device))

    # Quantize column by column with error compensation
    for i in range(W.shape[1]):
        w_col = W_gptq[:, i]
        scale = w_col.abs().max() / 127
        if scale == 0:
            scale = 1.0

        w_q = torch.clamp(torch.round(w_col / scale), -128, 127) * scale
        W_gptq_quantized[:, i] = w_q

        # Compute error
        error = w_col - w_q

        # Compensate error in remaining columns using Hessian
        if i < W.shape[1] - 1:
            compensation = torch.outer(H_inv[:, i] / H_inv[i, i], error)
            W_gptq[:, i+1:] -= compensation[:, None]

    Y_gptq = W_gptq_quantized @ X
    error_gptq = torch.mean((Y_original - Y_gptq) ** 2).item()
    print(f"   MSE: {error_gptq:.6f}")

    improvement = (error_naive - error_gptq) / error_naive * 100
    print(f"\n✓ GPTQ reduced error by {improvement:.1f}% compared to naive quantization!")

    print("\nKey GPTQ Features:")
    print("  1. Uses calibration data to guide quantization")
    print("  2. Quantizes weights in order, compensating errors")
    print("  3. Uses second-order information (Hessian)")
    print("  4. Achieves better accuracy than naive quantization")

    clear_gpu_memory()


def load_and_quantize_with_autogptq(model_name: str, bits: int = 4):
    """
    Load and quantize a model using auto-gptq library.

    Note: This is a demonstration. Full GPTQ quantization requires:
    1. Downloading a pre-quantized model, OR
    2. Quantizing from scratch (requires calibration dataset)
    """
    print(f"\n" + "=" * 70)
    print(f"AUTO-GPTQ: {bits}-bit Quantization")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        print(f"\nAttempting to load pre-quantized GPTQ model...")
        print("Note: This requires a model that has been pre-quantized with GPTQ")

        # Example of loading a pre-quantized model
        # Many models on HuggingFace are available in GPTQ format
        # Format usually: "TheBloke/{model-name}-GPTQ"

        gptq_model_name = "TheBloke/opt-125m-gptq"  # Example GPTQ model

        print(f"\nTrying to load: {gptq_model_name}")
        print("(This is for demonstration - model may not exist)")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoGPTQForCausalLM.from_quantized(
                gptq_model_name,
                device="cuda:0",
                use_triton=False,  # Use CUDA kernels instead of Triton
            )

            print("✓ GPTQ model loaded successfully!")
            return model, tokenizer

        except Exception as e:
            print(f"ℹ️  Could not load pre-quantized model: {e}")
            print("\nTo use GPTQ in practice:")
            print("  1. Find pre-quantized models on HuggingFace (search for 'GPTQ')")
            print("  2. Or quantize your own model using a calibration dataset")
            return None, None

    except ImportError:
        print("❌ auto-gptq not installed")
        print("Install with: pip install auto-gptq")
        return None, None


def demonstrate_gptq_quantization_process():
    """
    Demonstrate the GPTQ quantization process step-by-step.
    """
    print("\n" + "=" * 70)
    print("GPTQ QUANTIZATION PROCESS")
    print("=" * 70)

    print("\nStep-by-step GPTQ quantization:")

    print("\n1. Prepare Calibration Dataset:")
    print("   - Collect representative text samples (usually 128-1024 samples)")
    print("   - Examples: WikiText, C4, or domain-specific data")
    print("   - More diverse data = better quantization")

    print("\n2. Configure Quantization:")
    print("   - Bits: 4-bit, 3-bit, or 2-bit")
    print("   - Group size: 128 is common (smaller = better quality, more overhead)")
    print("   - Activation order: Reorder weights for better quantization")

    print("\n3. Run Quantization:")
    print("   - Process each layer sequentially")
    print("   - For each layer:")
    print("     a. Collect activations using calibration data")
    print("     b. Compute Hessian matrix from activations")
    print("     c. Quantize weights column-by-column")
    print("     d. Compensate errors using Hessian inverse")

    print("\n4. Save Quantized Model:")
    print("   - Save quantized weights and scales")
    print("   - Can be loaded for fast inference")

    print("\nExample configuration:")
    print("""
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,                    # 4-bit quantization
    group_size=128,            # Group size for quantization
    desc_act=False,            # Disable activation order (faster)
    damp_percent=0.01,         # Damping for numerical stability
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

# Quantize with calibration data
model.quantize(calibration_dataset)

# Save
model.save_quantized("./gptq-model")
    """)


def compare_quantization_methods():
    """
    Compare different quantization methods.
    """
    print("\n" + "=" * 70)
    print("QUANTIZATION METHODS COMPARISON")
    print("=" * 70)

    comparison_data = [
        {
            'method': 'FP16',
            'bits': 16,
            'memory': '1.0x',
            'speed': 'Baseline',
            'accuracy': '100%',
            'setup': 'Instant',
            'use_case': 'High accuracy required'
        },
        {
            'method': 'INT8 (Dynamic)',
            'bits': 8,
            'memory': '0.5x',
            'speed': '1.5-2x',
            'accuracy': '99-99.5%',
            'setup': 'Instant',
            'use_case': 'General inference'
        },
        {
            'method': 'INT8 (bitsandbytes)',
            'bits': 8,
            'memory': '0.25x',
            'speed': '1.2-1.5x',
            'accuracy': '98-99%',
            'setup': 'Instant',
            'use_case': 'Memory-constrained inference'
        },
        {
            'method': 'NF4 (bitsandbytes)',
            'bits': 4,
            'memory': '0.125x',
            'speed': '1.0-1.2x',
            'accuracy': '95-98%',
            'setup': 'Instant',
            'use_case': 'Maximum memory savings, QLoRA'
        },
        {
            'method': 'GPTQ',
            'bits': 4,
            'memory': '0.125x',
            'speed': '2-3x',
            'accuracy': '97-99%',
            'setup': '10-60 min',
            'use_case': 'Production inference (best quality/speed)'
        },
        {
            'method': 'AWQ',
            'bits': 4,
            'memory': '0.125x',
            'speed': '2-3x',
            'accuracy': '97-99%',
            'setup': '10-30 min',
            'use_case': 'Production inference (alternative to GPTQ)'
        },
    ]

    # Print table
    print("\n{:<20} {:<6} {:<10} {:<12} {:<10} {:<12} {:<30}".format(
        "Method", "Bits", "Memory", "Speed", "Accuracy", "Setup", "Best Use Case"
    ))
    print("-" * 110)

    for data in comparison_data:
        print("{:<20} {:<6} {:<10} {:<12} {:<10} {:<12} {:<30}".format(
            data['method'],
            data['bits'],
            data['memory'],
            data['speed'],
            data['accuracy'],
            data['setup'],
            data['use_case']
        ))

    print("\n\nMethod Details:")

    print("\n1. bitsandbytes (LLM.int8, NF4):")
    print("   Pros: Easy to use, no calibration needed, great for fine-tuning")
    print("   Cons: Slower inference than GPTQ/AWQ")

    print("\n2. GPTQ:")
    print("   Pros: Excellent accuracy, fast inference, widely supported")
    print("   Cons: Requires calibration, one-time setup cost")

    print("\n3. AWQ (Activation-aware Weight Quantization):")
    print("   Pros: Better accuracy than GPTQ for some models, fast inference")
    print("   Cons: Requires calibration, newer (less mature)")

    print("\n4. Dynamic INT8:")
    print("   Pros: Built into PyTorch, no setup, good accuracy")
    print("   Cons: Less memory savings (only weights quantized)")


def main():
    print("=" * 70)
    print("GPTQ Quantization Tutorial - Example 4")
    print("=" * 70 + "\n")

    # Check environment
    has_gpu = check_gpu()
    if not has_gpu:
        print("\n⚠️  Running without GPU. Some examples will be skipped.")

    # Part 1: Explain GPTQ concept
    if has_gpu:
        demonstrate_gptq_concept()

    # Part 2: Demonstrate quantization process
    demonstrate_gptq_quantization_process()

    # Part 3: Attempt to use auto-gptq
    if check_dependencies():
        model_name = "facebook/opt-125m"
        load_and_quantize_with_autogptq(model_name)

    # Part 4: Compare methods
    compare_quantization_methods()

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n✓ Key Takeaways:")
    print("  1. GPTQ uses calibration data for optimal quantization")
    print("  2. Achieves better accuracy than naive 4-bit quantization")
    print("  3. Requires one-time setup but gives fast inference")
    print("  4. Ideal for production deployments")

    print("\n✓ When to Use GPTQ:")
    print("  - Production inference with strict latency requirements")
    print("  - Need best 4-bit accuracy")
    print("  - Can afford one-time quantization cost")
    print("  - Have representative calibration data")

    print("\n✓ When to Use bitsandbytes Instead:")
    print("  - Quick experimentation")
    print("  - Fine-tuning with QLoRA")
    print("  - No calibration data available")
    print("  - Inference speed less critical")

    print("\n✓ Practical Resources:")
    print("  - Pre-quantized models: https://huggingface.co/TheBloke")
    print("  - Auto-GPTQ library: https://github.com/PanQiWei/AutoGPTQ")
    print("  - GPTQ paper: https://arxiv.org/abs/2210.17323")

    print("\n✓ Next Steps:")
    print("  - Try loading pre-quantized GPTQ models from HuggingFace")
    print("  - Experiment with different group sizes")
    print("  - Compare GPTQ vs bitsandbytes for your use case")
    print("  - Run Example 5 for comprehensive benchmarking")

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
