from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset():
    print(f"Looking for images in: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=TRANSFORM
    )

    return dataset


# ============================================================
# 5. SPLIT DATASET
# ============================================================

def split_dataset(dataset):

    total_size = len(dataset)

    if total_size == 0:
        raise ValueError(
            "No images found. Add the chest X-ray dataset to "
            "data/raw/normal, data/raw/pneumonia, and data/raw/covid."
        )

    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)

    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 7. CREATE RESNET-18 MODEL
# ============================================================

def create_model():

    model = models.resnet18(weights="DEFAULT")

    # Replace the original ImageNet classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 9. VALIDATE MODEL
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 10. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("MEDSCAN DIAGNOSTICS - RESNET-18 BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset()

    print(f"Total images: {len(dataset)}")

    print(f"Classes: {dataset.classes}")

    print(f"Class mapping: {dataset.class_to_idx}")

    # Check expected classes
    expected_classes = {
        "normal",
        "pneumonia",
        "covid"
    }

    actual_classes = set(dataset.classes)

    if actual_classes != expected_classes:

        raise ValueError(
            f"Expected classes {expected_classes}, "
            f"but found {actual_classes}"
        )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset
    )

    print()
    print("Dataset split:")
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(device)

    print()
    print("Model: ResNet-18")
    print(f"Number of classes: {NUM_CLASSES}")

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    history = []

    print()
    print("Starting training...")
    print()

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            model_path = MODEL_DIR / "resnet18_baseline.pth"

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"Best model saved: {model_path}"
            )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = REPORT_DIR / "baseline_training_history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy = validate(
        model,
        test_loader,
        criterion,
        device
    )

    print()
    print("=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print()
    print(f"Model saved in: {MODEL_DIR}")
    print(f"Training history: {history_path}")


# ============================================================
# 11. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
