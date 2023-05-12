""" Overall configuration. """
from pathlib import Path

import tensorflow as tf

import kitt

TF_VERSION = tf.__version__
AVAILABLE_GPU = tf.config.experimental.list_physical_devices("GPU")
NUM_AVAILABLE_GPU = len(AVAILABLE_GPU)
GPU_ACTIVE = bool(NUM_AVAILABLE_GPU)
EPSILON = 1e-6

BASE_PATH = Path(kitt.__file__).parent.parent
DATASET_DIR = BASE_PATH / "datasets"
BIN_DIR = BASE_PATH / "bin"
SAVES_DIR = BASE_PATH / "saves"
LOG_DIR = SAVES_DIR / "logs"
MODEL_SAVE_DIR = SAVES_DIR / "models"
PLOT_DIR = SAVES_DIR / "plots"
TABLES_DIR = SAVES_DIR / "tables"