from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .config import (
    RAW_DIR, SPLITS_DIR, CLASS_NAMES, CLASS_TO_INDEX,
    VAL_SIZE, TEST_SIZE, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS
)
from .preprocessing import train_transform, eval_transform


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_dataset():
    records = []

    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: missing directory: {class_dir}")
            continue

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    "image_path": str(path.relative_to(RAW_DIR)),
                    "label": class_name,
                    "label_id": CLASS_TO_INDEX[class_name],
                })

    if not records:
        raise FileNotFoundError(
            "No images found. Put an approved dataset into "
            "data/raw/normal, data/raw/pneumonia, and data/raw/covid."
        )

    return pd.DataFrame(records)


def create_splits(df):
    # First separate the test set.
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

    # Adjust validation proportion relative to the remaining data.
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=RANDOM_SEED,
        stratify=train_val["label_id"],
    )

    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    return train, val, test


def build_dataloaders():
    # ImageFolder expects exactly the class-folder structure used by this project.
    train_dataset = ImageFolder(RAW_DIR, transform=train_transform)
    eval_dataset = ImageFolder(RAW_DIR, transform=eval_transform)

    # This function is mainly for smoke testing. Training/evaluation use the
    # explicit split CSVs below to avoid accidentally training on all images.
    return train_dataset, eval_dataset


if __name__ == "__main__":
    df = scan_dataset()
    print("Total images:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    train, val, test = create_splits(df)

    print("\nSplit sizes:")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))
