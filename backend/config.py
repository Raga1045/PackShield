"""Central configuration module for DefektGuard.

Handles paths, hyper-parameters, hardware settings, and environment variables.
"""

import os
from pathlib import Path

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("DEFEKTGUARD_DATA_DIR", BACKEND_DIR / "data"))
MODEL_DIR = Path(os.getenv("DEFEKTGUARD_MODEL_DIR", BACKEND_DIR / "model"))
REPORTS_DIR = Path(os.getenv("DEFEKTGUARD_REPORTS_DIR", BACKEND_DIR / "reports"))

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
DEFAULT_MODEL_PATH = MODEL_DIR / "resnet34_model.keras"
FALLBACK_H5_MODEL_PATH = MODEL_DIR / "resnet34_model.h5"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_cnn_model.keras"

# Image & Data Configuration
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", "256"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH", "256"))
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
CLASS_NAMES = ["intact", "damaged"]
NUM_CLASSES = len(CLASS_NAMES)
POSITIVE_CLASS = "damaged"  # Positive class index 1 for precision/recall calculation

# Training Hyperparameters
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "15"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.15"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.15"))

# API & Security Configuration
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in {"true", "1", "yes"}

# Set environment defaults for TensorFlow / Keras reproducibility
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))

