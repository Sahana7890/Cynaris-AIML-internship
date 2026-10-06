from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_dataset():
    """
    Load chest X-ray images from:
    data/raw/normal
    data/raw/pneumonia
    data/raw/covid
    """

    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=TRANSFORM
    )

    return dataset


def create_dataloader(batch_size=32, shuffle=True):
    """
    Create a DataLoader for the chest X-ray dataset.
    """

    dataset = load_dataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return loader


if __name__ == "__main__":
    print("MedScan data pipeline")
    print(f"Dataset directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        print("ERROR: data/raw directory does not exist.")
    else:
        class_folders = ["normal", "pneumonia", "covid"]

        for folder in class_folders:
            folder_path = DATA_DIR / folder
            count = len(list(folder_path.glob("*"))) if folder_path.exists() else 0
            print(f"{folder}: {count} images")

        print("\nDataset images are required before training.")
