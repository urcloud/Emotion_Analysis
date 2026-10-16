import random
from pathlib import Path

import numpy as np
import torch


# =========================================================
# Random Seed
# =========================================================

SEED = 42


def set_seed(seed=SEED):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # CUDA 환경에서 재현성을 높이기 위한 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Dataset
# =========================================================

DATA_PATH = "dataset/emotion_recognitions_merged.csv"

TEXT_COL = "text"
LABEL_COL = "label"

NUM_CLASSES = 6


# =========================================================
# Data Split
# =========================================================

TRAIN_SIZE = 0.70
TEMP_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50


# =========================================================
# General Training
# =========================================================

EPOCHS = 10
BATCH_SIZE = 64

LEARNING_RATE = 1e-3


# =========================================================
# DataLoader
# =========================================================

NUM_WORKERS = 0


# =========================================================
# BoW
# =========================================================

MAX_FEATURES = 1000

BOW_HIDDEN_DIM = 256


# =========================================================
# Transformer from Scratch
# =========================================================

MAX_LEN = 100

EMBED_DIM = 128

NHEAD = 4

FFN_DIM = 256

NUM_LAYERS = 2

DROPOUT = 0.1


# =========================================================
# Pretrained Transformer
# =========================================================

PRETRAINED_MAX_LEN = 128

PRETRAINED_LR = 2e-5


# =========================================================
# Pretrained Model Checkpoints
# =========================================================

MODEL_NAMES = {

    "distilbert":
        "distilbert-base-uncased",

    "bert":
        "bert-base-uncased",

    "roberta":
        "roberta-base",

    "deberta":
        "microsoft/deberta-base"
}


# =========================================================
# Training Optimization
# =========================================================

# Learning Rate Experiment
LR_LIST = [
    1e-5,
    2e-5,
    5e-5
]


# Batch Size Experiment
BATCH_SIZE_LIST = [
    16,
    32,
    64
]


# Max Length Experiment
MAX_LENGTH_LIST = [
    64,
    128,
    256
]


# =========================================================
# Focal Loss
# =========================================================

FOCAL_GAMMA = 2.0


# =========================================================
# Data Augmentation
# =========================================================

AUGMENTATION_PROBABILITY = 0.1

AUGMENTATION_TYPE = "random_deletion"


# =========================================================
# Results
# =========================================================

RESULTS_DIR = Path("results")

CONFUSION_DIR = (
    RESULTS_DIR / "confusion_matrices"
)

ERROR_DIR = (
    RESULTS_DIR / "error_analysis"
)

ATTENTION_DIR = (
    RESULTS_DIR / "attention"
)

MODEL_DIR = (
    RESULTS_DIR / "checkpoints"
)


# =========================================================
# Result Files
# =========================================================

METRICS_PATH = (
    RESULTS_DIR / "metrics.csv"
)

EFFICIENCY_PATH = (
    RESULTS_DIR / "efficiency.csv"
)


# =========================================================
# Create Result Directories
# =========================================================

RESULTS_DIR.mkdir(
    exist_ok=True
)

CONFUSION_DIR.mkdir(
    exist_ok=True
)

ERROR_DIR.mkdir(
    exist_ok=True
)

ATTENTION_DIR.mkdir(
    exist_ok=True
)

MODEL_DIR.mkdir(
    exist_ok=True
)