"""Configuración compartida por train.py y evaluate.py."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = BASE_DIR / "data" / "requirements.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "requirements_test.csv"
MODEL_DIR = BASE_DIR / "model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

BASE_MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128

# El orden define los ids del modelo (id2label). Cambiarlo obliga a reentrenar.
LABEL_NAMES = [
    "FUNCTIONAL",
    "NON_FUNCTIONAL",
    "BUSINESS_RULE",
    "CONSTRAINT",
]


def normalize_text(text):
    """Forma canónica de un texto para detectar duplicados y fugas entre CSVs."""
    return " ".join(str(text).split()).lower()
