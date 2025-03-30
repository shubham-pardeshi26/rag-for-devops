import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")

import transformers
import torch

print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
