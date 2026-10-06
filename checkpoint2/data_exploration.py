"""MedScan Diagnostics - Checkpoint 2 data exploration.

Run from the project root:
    python data_exploration.py
"""

from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAMES = ["normal", "pneumonia", "covid"]
RAW_DIR = Path("data/raw")
FIGURES_DIR = Path("reports/figures")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_images():
    records = []
    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                records.append((class_name, path))
    return records


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    records = scan_images()

    counts = Counter(label for label, _ in records)

    print("\nMedScan Diagnostics - Dataset Exploration")
    print("-" * 50)
    print(f"Total images: {len(records)}")
    for class_name in CLASS_NAMES:
        print(f"{class_name:12}: {counts[class_name]}")

    if not records:
        print("\nNo X-ray images found.")
        print("Place approved images in:")
        print("  data/raw/normal/")
        print("  data/raw/pneumonia/")
        print("  data/raw/covid/")
        return

    plt.figure(figsize=(7, 5))
    plt.bar(CLASS_NAMES, [counts[c] for c in CLASS_NAMES])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Show up to one sample from each class.
    samples = []
    for class_name in CLASS_NAMES:
        class_samples = [p for label, p in records if label == class_name]
        if class_samples:
            samples.append((class_name, class_samples[0]))

    fig, axes = plt.subplots(1, len(samples), figsize=(12, 4))
    if len(samples) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, samples):
        try:
            image = Image.open(path).convert("RGB")
            ax.imshow(image)
            ax.set_title(label)
            ax.axis("off")
        except Exception as exc:
            ax.set_title(f"{label}\nError: {exc}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sample_images.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
