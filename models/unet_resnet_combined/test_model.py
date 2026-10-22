
import torch
from model import create_model


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = create_model().to(device)
    dummy_input = torch.randn(2, 5, 352, 352).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)

    assert output.shape == (2, 1, 352, 352)
    print("Combined model test passed!")


if __name__ == "__main__":
    main()
