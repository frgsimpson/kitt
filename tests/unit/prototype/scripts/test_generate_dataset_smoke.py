""" Test :mod kitt.prototype.scripts.generate_dataset does not smoke """
import argparse
import time
from pathlib import Path

import pytest

from kitt.config import DATASET_DIR
from kitt.prototype.scripts.generate_dataset import main as generate_dataset


@pytest.mark.parametrize("num_dims", (1, 2, 3))
@pytest.mark.parametrize("include_x_with_samples", (True, False))
@pytest.mark.parametrize("make_captions", (True, False))
def test_generate_dataset__does_not_smoke(
    tmp_path: Path,
    monkeypatch,
    num_dims: int,
    make_captions: bool,
    include_x_with_samples: bool,
):
    tmp_name = f"tmp-{time.time()}"

    mock_args = argparse.Namespace(
        num_dimensions=num_dims,
        min_expression=1,
        max_expression=1,
        train_sigma=1.,
        test_sigma=1.,
        num_expressions=10,
        train_ratio=.8,
        num_samples_per_expression=2,
        num_hyper_randomisations=1,
        max_products=1,
        make_captions=make_captions,
        num_channels=0,
        output_dir=tmp_name,
        include_x_with_samples=include_x_with_samples,
        use_fft=False,
        random_seed=42,
    )

    try:
        generate_dataset(mock_args)
    except:
        pytest.fail("Could not run generate_dataset")
    finally:
        (DATASET_DIR / (tmp_name + ".tar.gz")).unlink()


def test_generate_dataset_for_transformer__does_not_smoke(tmp_path: Path, monkeypatch):
    tmp_name = f"tmp-{time.time()}"
    mock_args = argparse.Namespace(
        num_dimensions=4,
        min_expression=1,
        max_expression=1,
        train_sigma=1.0,
        test_sigma=1.0,
        num_expressions=10,
        train_ratio=0.8,
        num_samples_per_expression=2,
        num_hyper_randomisations=1,
        max_products=1,
        make_captions=True,
        num_channels=0,
        output_dir=tmp_name,
        include_x_with_samples=True,
        use_fft=False,
        random_seed=42,
    )

    try:
        generate_dataset(mock_args)
    except:
        pytest.fail("Could not run generate_dataset")
    finally:
        (DATASET_DIR / (tmp_name + ".tar.gz")).unlink()
