""" Test :mod kitt.prototype.scripts.train_kitt does not smoke """
import argparse
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from kitt.config import DATASET_DIR
from kitt.prototype.scripts.train_kitt import main as train_and_evaluate


@pytest.mark.parametrize("decoder", ["decoder-rnn", "decoder-transformer"])
def test_train_and_evaluate__does_not_smoke(tmp_path: Path, decoder: str):

    with patch("kitt.prototype.scripts.train_kitt.get_arguments") as mock_args:
        mock_args.return_value = argparse.Namespace(
            pretrained_encoder=None,
            epochs=1,
            steps_per_epoch=2,
            save_model=False,
            lr=1e-5,
            iters_per_update=2,
            batch_size=16,
            dataset=None,  # todo support running with preset dataset
            test_sigma=1.,
            plot=False,
            resolution=64,
            num_dimensions=2,
            max_caption_length=3,
            decoder_name=decoder,
            pretrained_encoder_lr=None
        )

        try:
            train_and_evaluate()
        except:
            pytest.fail("Could not run training")
        finally:
            # remove training byproduct
            checkpoint_dir = DATASET_DIR.parent / "checkpoints"
            if checkpoint_dir.exists():
                shutil.rmtree(str(checkpoint_dir), ignore_errors=True)
