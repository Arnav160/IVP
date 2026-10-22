import torch

from dataloader import get_dataloaders


def check_loader(loader, name):

    print("\n" + "=" * 60)
    print(f"TESTING {name}")
    print("=" * 60)

    images, masks = next(iter(loader))

    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")

    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")

    print(
        f"Mask unique values: "
        f"{torch.unique(masks)}"
    )


def main():

    print("=" * 60)
    print("COD10K DATALOADER TEST")
    print("=" * 60)

    train_loader, val_loader, test_loader = (
        get_dataloaders()
    )

    check_loader(
        train_loader,
        "TRAIN LOADER"
    )

    check_loader(
        val_loader,
        "VALIDATION LOADER"
    )

    check_loader(
        test_loader,
        "TEST LOADER"
    )

    print("\n" + "=" * 60)
    print("DATALOADER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()