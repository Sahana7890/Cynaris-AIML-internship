from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CLASS_NAMES = ["normal", "pneumonia", "covid"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_CLASSES = len(CLASS_NAMES)

VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4

MODEL_PATH = MODELS_DIR / "resnet18_baseline.pth"

for directory in [SPLITS_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
