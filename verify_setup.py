"""
Setup Verification Script

Run this script to verify your environment is properly configured
for the LLM Quantization Tutorial.

Usage: python verify_setup.py
"""

import sys


def check_python_version():
    """Check Python version."""
    print("\n" + "=" * 70)
    print("PYTHON VERSION")
    print("=" * 70)

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "=" * 70)
    print("PYTORCH")
    print("=" * 70)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")

        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")

            # GPU info
            num_gpus = torch.cuda.device_count()
            print(f"  Number of GPUs: {num_gpus}")

            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                compute = torch.cuda.get_device_properties(i).major
                print(f"  GPU {i}: {name}")
                print(f"    Memory: {memory:.1f} GB")
                print(f"    Compute Capability: {compute}.{torch.cuda.get_device_properties(i).minor}")

                if memory < 4:
                    print("    ⚠️  Warning: Less than 4GB VRAM. Some examples may not run.")

            return True
        else:
            print("  ⚠️  Warning: CUDA not available. Tutorial will run on CPU (slower).")
            print("     Examples 1-2 will work, but 3-5 may be very slow or fail.")
            return True

    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return False


def check_transformers():
    """Check transformers installation."""
    print("\n" + "=" * 70)
    print("TRANSFORMERS")
    print("=" * 70)

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")

        # Check version
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        if major < 4 or (major == 4 and minor < 35):
            print("  ⚠️  Version 4.35+ recommended")

        return True
    except ImportError:
        print("✗ Transformers not installed (required for Examples 3-5)")
        print("  Install with: pip install transformers")
        return False


def check_bitsandbytes():
    """Check bitsandbytes installation."""
    print("\n" + "=" * 70)
    print("BITSANDBYTES")
    print("=" * 70)

    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes {bnb.__version__}")
        return True
    except ImportError:
        print("✗ bitsandbytes not installed (required for Example 3)")
        print("  Install with: pip install bitsandbytes")
        print("  Note: Linux/WSL2 only, not available on Mac or native Windows")
        return False


def check_accelerate():
    """Check accelerate installation."""
    print("\n" + "=" * 70)
    print("ACCELERATE")
    print("=" * 70)

    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
        return True
    except ImportError:
        print("✗ Accelerate not installed (required for Examples 3-5)")
        print("  Install with: pip install accelerate")
        return False


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\n" + "=" * 70)
    print("OPTIONAL DEPENDENCIES")
    print("=" * 70)

    optional = {}

    # auto-gptq
    try:
        import auto_gptq
        print(f"✓ auto-gptq {auto_gptq.__version__} (for Example 4)")
        optional['auto-gptq'] = True
    except ImportError:
        print("○ auto-gptq not installed (optional, for Example 4)")
        print("  Install with: pip install auto-gptq")
        optional['auto-gptq'] = False

    # matplotlib
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__} (for visualizations)")
        optional['matplotlib'] = True
    except ImportError:
        print("○ matplotlib not installed (optional, for plots)")
        print("  Install with: pip install matplotlib")
        optional['matplotlib'] = False

    # numpy
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
        optional['numpy'] = True
    except ImportError:
        print("✗ numpy not installed (required)")
        print("  Install with: pip install numpy")
        optional['numpy'] = False

    return optional


def test_gpu_computation():
    """Test GPU computation."""
    print("\n" + "=" * 70)
    print("GPU COMPUTATION TEST")
    print("=" * 70)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⊘ Skipping (no CUDA)")
            return True

        print("Running simple GPU computation test...")

        # Create tensors
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        # Computation
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        print("✓ GPU computation successful")

        # Memory test
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  Memory allocated: {allocated:.2f} GB")
        print(f"  Memory reserved: {reserved:.2f} GB")

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        return False


def print_summary(results):
    """Print summary of verification results."""
    print("\n\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    essential = {
        'Python 3.8+': results['python'],
        'PyTorch': results['pytorch'],
    }

    required_for_full = {
        'Transformers': results['transformers'],
        'Accelerate': results['accelerate'],
        'bitsandbytes': results['bitsandbytes'],
    }

    print("\nEssential Components:")
    for name, status in essential.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")

    print("\nRequired for Full Tutorial:")
    for name, status in required_for_full.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")

    print("\nOptional Components:")
    for name, status in results.get('optional', {}).items():
        symbol = "✓" if status else "○"
        print(f"  {symbol} {name}")

    # Overall status
    print("\n" + "=" * 70)

    all_essential = all(essential.values())
    all_required = all(required_for_full.values())

    if all_essential and all_required:
        print("✓ SETUP COMPLETE - Ready to run all examples!")
        print("\nNext steps:")
        print("  1. Read README.md for tutorial overview")
        print("  2. Run: python examples/01_basic_int8_quantization.py")
        print("  3. Progress through examples in order")

    elif all_essential and results['pytorch']:
        print("⚠️  PARTIAL SETUP - Can run basic examples (1-2)")
        print("\nMissing dependencies for advanced examples (3-5).")
        print("Install with: pip install -r requirements.txt")

    else:
        print("✗ SETUP INCOMPLETE - Please install missing dependencies")
        print("\nSee SETUP.md for detailed installation instructions")

    print("=" * 70)


def main():
    """Main verification routine."""
    print("=" * 70)
    print("LLM Quantization Tutorial - Setup Verification")
    print("=" * 70)

    results = {}

    # Check all components
    results['python'] = check_python_version()
    results['pytorch'] = check_pytorch()
    results['transformers'] = check_transformers()
    results['bitsandbytes'] = check_bitsandbytes()
    results['accelerate'] = check_accelerate()
    results['optional'] = check_optional_dependencies()

    if results['pytorch']:
        results['gpu_test'] = test_gpu_computation()

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
