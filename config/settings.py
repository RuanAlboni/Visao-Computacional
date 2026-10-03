"""
Configurações globais do projeto Smart Fridge Vision
"""
from pathlib import Path

# Caminhos base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# Datasets
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Modelos
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
SVM_MODELS_DIR = MODELS_DIR / "svm"

# Imagens
IMAGE_SIZE = 640  # Tamanho para YOLO
CONFIDENCE_THRESHOLD = 0.5  # Threshold para detecção
IOU_THRESHOLD = 0.45  # NMS threshold

# Classes
CLASSES = [
    "tomato_appropriate",
    "tomato_inappropriate",
    "apple_appropriate",
    "apple_inappropriate",
    "strawberry_appropriate",
    "strawberry_inappropriate"
]

# Device
DEVICE = "cuda"  # ou "cpu" se não tiver GPU