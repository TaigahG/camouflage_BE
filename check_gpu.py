"""
Diagnostic script to check GPU availability and PyTorch configuration
"""
import torch
import sys

print("=" * 60)
print("GPU DETECTION DIAGNOSTIC")
print("=" * 60)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ Python version: {sys.version}")

print("\n--- CUDA Support ---")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA device properties:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {props.name} - {props.total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA is NOT available")
    print("This could mean:")
    print("  - PyTorch was installed without CUDA support (CPU-only)")
    print("  - NVIDIA drivers are not installed")
    print("  - CUDA toolkit is not installed")

print("\n--- MPS Support (Apple Metal) ---")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print(f"MPS built: {torch.backends.mps.is_built()}")

print("\n--- CPU Info ---")
print(f"CPU count: {torch.get_num_threads()}")

print("\n--- Recommended Device ---")
if torch.cuda.is_available():
    print("✅ GPU (CUDA) - RECOMMENDED")
elif torch.backends.mps.is_available():
    print("✅ MPS (Apple Metal) - RECOMMENDED")
else:
    print("⚠️  CPU only - Performance will be limited")

print("\n" + "=" * 60)
