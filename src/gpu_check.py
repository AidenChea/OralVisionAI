import torch

print("=" * 40)
print("OralVision AI")
print("=" * 40)

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y

    print("Tensor device:", z.device)
    print("✅ GPU test passed!")
else:
    print("❌ CUDA is not available.")
    