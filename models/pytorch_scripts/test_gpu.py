import torch

from dataloader import get_dataloaders


def main():

    print("=" * 60)
    print("COD10K GPU TEST")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():

        print("\nCUDA is NOT available.")
        print("GPU test cannot continue.")

        return

    device = torch.device("cuda")

    print(f"\nGPU available: YES")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")

    # Load dataloaders
    train_loader, _, _ = get_dataloaders()

    # Get one batch
    images, masks = next(iter(train_loader))

    print("\nBefore GPU transfer:")

    print(f"Images device: {images.device}")
    print(f"Masks device:  {masks.device}")

    # Move batch to GPU
    images = images.to(
        device,
        non_blocking=True
    )

    masks = masks.to(
        device,
        non_blocking=True
    )

    print("\nAfter GPU transfer:")

    print(f"Images device: {images.device}")
    print(f"Masks device:  {masks.device}")

    print(f"\nImage batch shape: {images.shape}")
    print(f"Mask batch shape:  {masks.shape}")

    print("\n" + "=" * 60)
    print("GPU TRANSFER TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()