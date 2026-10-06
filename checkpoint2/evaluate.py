import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    RAW_DIR, SPLITS_DIR, MODEL_PATH, FIGURES_DIR,
    BATCH_SIZE, NUM_WORKERS, CLASS_NAMES
)
from .preprocessing import eval_transform
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
        return self.transform(image), int(row["label_id"])


def main():
    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Run dataset splitting and training before evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = CSVDataset(test_csv, eval_transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = y_prob.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    # One-vs-rest multiclass ROC-AUC.
    y_onehot = np.eye(len(CLASS_NAMES))[y_true]
    auc_macro = roc_auc_score(
        y_onehot, y_prob, multi_class="ovr", average="macro"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # COVID is class index 2.
    covid_tp = cm[2, 2]
    covid_fn = cm[2, :].sum() - covid_tp
    covid_fnr = covid_fn / (covid_tp + covid_fn) if (covid_tp + covid_fn) else np.nan

    print("\n=== BASELINE TEST RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro ROC-AUC (OvR): {auc_macro:.4f}")
    print(f"Normal    - precision={precision[0]:.4f}, recall={recall[0]:.4f}, f1={f1[0]:.4f}")
    print(f"Pneumonia - precision={precision[1]:.4f}, recall={recall[1]:.4f}, f1={f1[1]:.4f}")
    print(f"COVID     - precision={precision[2]:.4f}, recall={recall[2]:.4f}, f1={f1[2]:.4f}")
    print(f"COVID FNR: {covid_fnr:.4f}")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print("Saved confusion matrix to:", FIGURES_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
