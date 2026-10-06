import copy
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from .config import (
    RAW_DIR, SPLITS_DIR, MODEL_PATH, BATCH_SIZE, NUM_WORKERS,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, RANDOM_SEED
)
from .preprocessing import train_transform, eval_transform
from .model import create_model


class CSVDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(RAW_DIR / row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label_id"])
        return image, label


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    set_seed()

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Split files not found. Run `python -m src.dataset` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = CSVDataset(train_csv, train_transform)
    val_ds = CSVDataset(val_csv, eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = create_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["normal", "pneumonia", "covid"],
        },
        MODEL_PATH,
    )

    print("Saved baseline model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
