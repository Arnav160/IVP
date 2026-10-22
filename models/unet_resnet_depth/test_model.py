
import torch
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = create_model().to(device)
model.eval()

# Simulate a batch of two RGB + depth images
dummy_input = torch.randn(2, 4, 352, 352, device=device)

with torch.inference_mode():
    output = model(dummy_input)

print("Input shape:", tuple(dummy_input.shape))
print("Output shape:", tuple(output.shape))
print("Output dtype:", output.dtype)

assert output.shape == (2, 1, 352, 352)
print("Model test passed!")
