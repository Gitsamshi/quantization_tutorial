"""
Example 3: Real LLM Quantization with bitsandbytes

This script demonstrates real LLM quantization using the bitsandbytes library:
- Loading models in 8-bit and 4-bit precision
- Memory usage comparison
- Inference speed benchmarking
- Quality assessment

Run: python examples/03_llm_bitsandbytes.py

Note: This example downloads a real language model (~2-3GB). Make sure you have:
- Sufficient disk space
- Good internet connection
- At least 8GB GPU VRAM (for larger models)
"""

import torch
import time
from typing import Dict, List
import os


def check_dependencies():
    """Check if required libraries are installed."""
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return False

    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("❌ bitsandbytes not installed. Install with: pip install bitsandbytes")
        return False

    try:
        import accelerate
        print(f"✓ accelerate version: {accelerate.__version__}")
    except ImportError:
        print("❌ accelerate not installed. Install with: pip install accelerate")
        return False

    return True


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a GPU.")
        return False

    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ Total GPU memory: {total_memory:.2f} GB")

    if total_memory < 6:
        print("⚠️  Warning: Less than 6GB GPU memory. Consider using a smaller model.")

    return True


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_fp16(model_name: str):
    """Load model in FP16 precision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name} in FP16...")
    clear_gpu_memory()
    start_mem = get_gpu_memory_usage()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    end_mem = get_gpu_memory_usage()
    memory_used = end_mem - start_mem

    return model, tokenizer, memory_used


def load_model_8bit(model_name: str):
    """Load model in 8-bit precision using bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name} in INT8 (bitsandbytes)...")
    clear_gpu_memory()
    start_mem = get_gpu_memory_usage()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )

    end_mem = get_gpu_memory_usage()
    memory_used = end_mem - start_mem

    return model, tokenizer, memory_used


def load_model_4bit(model_name: str):
    """Load model in 4-bit precision using bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading {model_name} in NF4 (bitsandbytes)...")
    clear_gpu_memory()
    start_mem = get_gpu_memory_usage()

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for better performance
        bnb_4bit_use_double_quant=True,  # Nested quantization for additional memory savings
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    end_mem = get_gpu_memory_usage()
    memory_used = end_mem - start_mem

    return model, tokenizer, memory_used


def benchmark_inference(model, tokenizer, prompt: str, num_runs: int = 5) -> Dict:
    """Benchmark model inference speed."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    tokens_generated = 50
    tokens_per_sec = tokens_generated / avg_time

    return {
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'output': tokenizer.decode(outputs[0], skip_special_tokens=True)
    }


def compare_outputs(outputs: Dict[str, str]) -> None:
    """Compare outputs from different quantization levels."""
    print("\n" + "=" * 70)
    print("OUTPUT COMPARISON")
    print("=" * 70)

    for precision, output in outputs.items():
        print(f"\n{precision}:")
        print("-" * 70)
        print(output[:500])  # Print first 500 chars
        if len(output) > 500:
            print("...")


def main():
    print("=" * 70)
    print("Real LLM Quantization with bitsandbytes - Example 3")
    print("=" * 70 + "\n")

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install required packages.")
        return

    if not check_gpu():
        print("\n❌ GPU required for this example.")
        return

    # ========================================================================
    # Configuration
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    # Use a smaller model for demonstration
    # Options: "gpt2", "gpt2-medium", "facebook/opt-125m", "facebook/opt-350m"
    model_name = "facebook/opt-350m"  # ~350M parameters, ~700MB in FP16

    print(f"\nModel: {model_name}")
    print("Note: First run will download the model. Subsequent runs will use cached version.")

    test_prompt = "The future of artificial intelligence is"
    print(f"\nTest prompt: '{test_prompt}'")

    # ========================================================================
    # Part 1: FP16 Baseline
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 1: FP16 Baseline")
    print("=" * 70)

    try:
        model_fp16, tokenizer, mem_fp16 = load_model_fp16(model_name)
        print(f"✓ Model loaded successfully")
        print(f"✓ GPU memory used: {mem_fp16:.2f} GB")

        print("\nRunning inference benchmark...")
        results_fp16 = benchmark_inference(model_fp16, tokenizer, test_prompt)
        print(f"✓ Average time: {results_fp16['avg_time']:.3f} seconds")
        print(f"✓ Throughput: {results_fp16['tokens_per_sec']:.1f} tokens/sec")

        # Clean up
        del model_fp16
        clear_gpu_memory()

    except Exception as e:
        print(f"❌ Error loading FP16 model: {e}")
        print("This might be due to insufficient GPU memory.")
        mem_fp16 = None
        results_fp16 = None

    # ========================================================================
    # Part 2: INT8 Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: INT8 Quantization (bitsandbytes)")
    print("=" * 70)

    try:
        model_8bit, tokenizer, mem_8bit = load_model_8bit(model_name)
        print(f"✓ Model loaded successfully")
        print(f"✓ GPU memory used: {mem_8bit:.2f} GB")

        if mem_fp16:
            reduction = mem_fp16 / mem_8bit
            savings = ((mem_fp16 - mem_8bit) / mem_fp16) * 100
            print(f"✓ Memory reduction: {reduction:.2f}x ({savings:.1f}% savings)")

        print("\nRunning inference benchmark...")
        results_8bit = benchmark_inference(model_8bit, tokenizer, test_prompt)
        print(f"✓ Average time: {results_8bit['avg_time']:.3f} seconds")
        print(f"✓ Throughput: {results_8bit['tokens_per_sec']:.1f} tokens/sec")

        if results_fp16:
            speedup = results_8bit['tokens_per_sec'] / results_fp16['tokens_per_sec']
            print(f"✓ Speedup vs FP16: {speedup:.2f}x")

        # Clean up
        del model_8bit
        clear_gpu_memory()

    except Exception as e:
        print(f"❌ Error loading INT8 model: {e}")
        mem_8bit = None
        results_8bit = None

    # ========================================================================
    # Part 3: NF4 Quantization
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PART 3: NF4 (4-bit) Quantization (bitsandbytes)")
    print("=" * 70)

    try:
        model_4bit, tokenizer, mem_4bit = load_model_4bit(model_name)
        print(f"✓ Model loaded successfully")
        print(f"✓ GPU memory used: {mem_4bit:.2f} GB")

        if mem_fp16:
            reduction = mem_fp16 / mem_4bit
            savings = ((mem_fp16 - mem_4bit) / mem_fp16) * 100
            print(f"✓ Memory reduction: {reduction:.2f}x ({savings:.1f}% savings)")

        print("\nRunning inference benchmark...")
        results_4bit = benchmark_inference(model_4bit, tokenizer, test_prompt)
        print(f"✓ Average time: {results_4bit['avg_time']:.3f} seconds")
        print(f"✓ Throughput: {results_4bit['tokens_per_sec']:.1f} tokens/sec")

        if results_fp16:
            speedup = results_4bit['tokens_per_sec'] / results_fp16['tokens_per_sec']
            print(f"✓ Speedup vs FP16: {speedup:.2f}x")

        # Clean up
        del model_4bit
        clear_gpu_memory()

    except Exception as e:
        print(f"❌ Error loading NF4 model: {e}")
        mem_4bit = None
        results_4bit = None

    # ========================================================================
    # Part 4: Summary and Comparison
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY: Quantization Comparison")
    print("=" * 70)

    print("\n{:<15} {:<15} {:<20} {:<15}".format(
        "Precision", "Memory (GB)", "Speed (tokens/s)", "Memory Savings"
    ))
    print("-" * 70)

    if mem_fp16 and results_fp16:
        print("{:<15} {:<15.2f} {:<20.1f} {:<15}".format(
            "FP16", mem_fp16, results_fp16['tokens_per_sec'], "Baseline"
        ))

    if mem_8bit and results_8bit:
        savings_8bit = ((mem_fp16 - mem_8bit) / mem_fp16 * 100) if mem_fp16 else 0
        print("{:<15} {:<15.2f} {:<20.1f} {:<15}".format(
            "INT8", mem_8bit, results_8bit['tokens_per_sec'], f"{savings_8bit:.1f}%"
        ))

    if mem_4bit and results_4bit:
        savings_4bit = ((mem_fp16 - mem_4bit) / mem_fp16 * 100) if mem_fp16 else 0
        print("{:<15} {:<15.2f} {:<20.1f} {:<15}".format(
            "NF4", mem_4bit, results_4bit['tokens_per_sec'], f"{savings_4bit:.1f}%"
        ))

    # Compare output quality
    outputs = {}
    if results_fp16:
        outputs['FP16'] = results_fp16['output']
    if results_8bit:
        outputs['INT8'] = results_8bit['output']
    if results_4bit:
        outputs['NF4'] = results_4bit['output']

    if len(outputs) > 1:
        compare_outputs(outputs)

    # ========================================================================
    # Part 5: Advanced Features
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("ADVANCED FEATURES")
    print("=" * 70)

    print("\n1. Double Quantization:")
    print("   - Quantizes the quantization constants themselves")
    print("   - Saves additional ~0.4 bits per parameter")
    print("   - Enabled with: bnb_4bit_use_double_quant=True")

    print("\n2. Compute Dtype:")
    print("   - Weights stored in 4-bit, but computation in FP16/BF16")
    print("   - Better accuracy with minimal memory overhead")
    print("   - Set with: bnb_4bit_compute_dtype=torch.float16")

    print("\n3. Mixed Precision:")
    print("   - Keep some layers in higher precision")
    print("   - Useful for layers sensitive to quantization")
    print("   - Configure with custom quantization configs")

    # ========================================================================
    # Key Takeaways
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)

    print("\n✓ Practical Benefits:")
    print("  1. Significant memory savings (2-4x) with minimal quality loss")
    print("  2. Enables running larger models on consumer GPUs")
    print("  3. Bitsandbytes handles all complexity automatically")
    print("  4. Compatible with Hugging Face ecosystem")

    print("\n✓ Best Practices:")
    print("  1. Use INT8 for inference when memory is limited")
    print("  2. Use NF4 for maximum memory savings (QLoRA fine-tuning)")
    print("  3. Always enable double quantization for 4-bit")
    print("  4. Set compute_dtype to FP16/BF16 for better accuracy")

    print("\n✓ Use Cases:")
    print("  - INT8: Production inference, API serving")
    print("  - NF4: Fine-tuning with QLoRA, research, experimentation")
    print("  - Mixed: Keep attention layers in FP16, rest in 4-bit")

    print("\n✓ Next Steps:")
    print("  - Try with larger models (e.g., 'facebook/opt-1.3b')")
    print("  - Experiment with different quantization configs")
    print("  - Run Example 4 for GPTQ quantization")
    print("  - Try fine-tuning with QLoRA (Parameter Efficient Fine-Tuning)")

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Suppress some warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
