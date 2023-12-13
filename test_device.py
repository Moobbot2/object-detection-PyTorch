import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    # Set the device to CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
