from argparse import Namespace

from kitt.prototype.configs.configs import DATASET_GENERATION_CONFIGS, load_config
from kitt.prototype.scripts.generate_dataset import main as generate_dataset

if __name__ == "__main__":
    for config_name in DATASET_GENERATION_CONFIGS:
        config = load_config(config_name)
        generate_dataset(Namespace(**config))
