"""
Smoke-test configuration for the AD-vs-CN EEG connectivity CNN pipeline.

This configuration runs the complete pipeline with a very small CV/search
setup so that the code, file paths, training, evaluation, plotting,
checkpointing, and resume functionality can be tested before launching
the full experiment.

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

# Separate output directory so the smoke test cannot contaminate
# the results of the real experiment.
OUTPUT_DIR = PROJECT_DIR / "output_smoke_test"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

GLOBAL_SEED = 42


# ---------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------

POSITIVE_CLASS_LABEL = "AD"
NEGATIVE_CLASS_LABEL = "CN"

CLASS_TO_INT = {
    NEGATIVE_CLASS_LABEL: 0,
    POSITIVE_CLASS_LABEL: 1,
}

CLASS_NAMES = [
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
]


# ---------------------------------------------------------------------
# Cross-validation structure
# ---------------------------------------------------------------------

# Reduced only for the smoke test.
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5


# ---------------------------------------------------------------------
# Run / resume settings
# ---------------------------------------------------------------------

RESUME = True


# ---------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------

# configuration during the smoke test.
CONV_CONFIGS = [
    [(16, 3), (32, 3)]
]

FC_UNITS = [128]

DROPOUT_VALUES = [0.3]
LEARNING_RATES = [1e-4]
BATCH_SIZES = [16]

USE_POOLING = True      # MaxPool2d(2) after every conv block; see model.py.
                        # With INPUT_SIZE=19, safe for up to 3 conv blocks
                        # (19->9->4->2). A 4th block would hit 1x1 and raise
                        # an assertion error in EEGConnectivityCNN.__init__.

N_RANDOM_CONFIGS = 1


# ---------------------------------------------------------------------
# Fixed architecture settings
# ---------------------------------------------------------------------

N_CLASSES = 2
IN_CHANNELS = 5
INPUT_SIZE = 19


# ---------------------------------------------------------------------
# Fixed optimizer / training settings
# ---------------------------------------------------------------------

OPTIMIZER = "adamw"
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Short training for smoke test.
MAX_EPOCHS = 100

# Short patience for smoke test.
EARLY_STOPPING_PATIENCE = 6

# Short scheduler patience for smoke test.
LR_SCHEDULER_PATIENCE = 1
LR_SCHEDULER_FACTOR = 0.5


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

# Keep normalization enabled because the smoke test should exercise
# the actual preprocessing pipeline too.
NORMALIZE_PER_BAND = True


# ---------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------

NUM_WORKERS = 0
PIN_MEMORY = False


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = "cuda"


# ---------------------------------------------------------------------
# Quick baseline diagnostic
# ---------------------------------------------------------------------

BASELINE_FOLDS = 5