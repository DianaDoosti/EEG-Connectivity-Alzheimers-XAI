"""
Central configuration for the AD-vs-CN EEG connectivity CNN pipeline.

Every tunable and fixed setting lives here so the whole experiment is
reproducible from a single file. Edit this file, not the pipeline code,
to change the experiment.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

DATA_DIR = PROJECT_DIR / "eeg_connectivity_dataset"/ "concatenated"
DATA_FILENAME = "all_subjects_connectivity.npz"
DATA_PATH = DATA_DIR / DATA_FILENAME

OUTPUT_DIR = PROJECT_DIR / "output"

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
GLOBAL_SEED = 42  # controls: fold splits, HP-config sampling, weight init,
                   # dataloader shuffling, all torch/numpy/python RNGs

# ---------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------
POSITIVE_CLASS_LABEL = "AD"
NEGATIVE_CLASS_LABEL = "CN"
CLASS_TO_INT = {NEGATIVE_CLASS_LABEL: 0, POSITIVE_CLASS_LABEL: 1}
CLASS_NAMES = [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL]  # index 0, 1

# ---------------------------------------------------------------------
# Cross-validation structure
# ---------------------------------------------------------------------
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

# ---------------------------------------------------------------------
# Run / resume settings
# ---------------------------------------------------------------------
RESUME = True

# ---------------------------------------------------------------------
# Hyperparameter search space 
# ---------------------------------------------------------------------
# Conv configs: list of (out_channels, kernel_size) per conv block.
CONV_CONFIGS = [
    [(16, 3), (32, 3)],
    [(32, 3), (64, 3)],
    [(16, 3), (32, 3), (64, 3)],
    [(32, 3), (64, 3), (128, 3)],
]

# Width of the single FC hidden layer before the output layer.
FC_UNITS = [64, 128]

DROPOUT_VALUES = [0.2, 0.3, 0.5]
LEARNING_RATES = [1e-3, 1e-4]  
BATCH_SIZES = [16]

USE_POOLING = True      # MaxPool2d(2) after every conv block; see model.py.
                        
N_RANDOM_CONFIGS = 24

# ---------------------------------------------------------------------
# Fixed architecture settings (not tuned)
# ---------------------------------------------------------------------
N_CLASSES = 2           # softmax output (2 logits), matches weighted CE loss
IN_CHANNELS = 5         # 5 frequency bands as input channels
INPUT_SIZE = 19         # 19x19 connectivity matrix

# ---------------------------------------------------------------------
# Fixed optimizer / training settings (not tuned)
# ---------------------------------------------------------------------
OPTIMIZER = "adamw"
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 6      # epochs with no val-loss improvement
LR_SCHEDULER_PATIENCE = 2        # ReduceLROnPlateau patience
LR_SCHEDULER_FACTOR = 0.5

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
NORMALIZE_PER_BAND = True

# ---------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------
NUM_WORKERS = 0   # 0 avoids Windows multiprocessing issues 
PIN_MEMORY = False

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
DEVICE = "cuda"  # falls back to "cpu" automatically at runtime if unavailable