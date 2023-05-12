""" Test for :mod kitt.data.sampler.sample_generation """
from typing import Optional

import gpflow
import numpy as np
import pytest

from kitt.data.kernels import KernelType, get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import load_default_coords
from kitt.data.tokeniser import KernelTokenizer

SINGLE_EXPRESSION = [KernelType.LINEAR.name]
COMPOUND_EXPRESSION = [KernelType.LINEAR.name, KernelType.MATERN52.name, KernelType.RBF.name]


@pytest.fixture(scope="module", name="tokeniser")
def _tokeniser() -> KernelTokenizer:
    return KernelTokenizer()


def get_sampler(
    resolution: int = 28,
    min_expression: int = 1,
    max_expression: int = 1,
    make_captions: bool = True,
    use_fft: bool = True,
    tokeniser: Optional[KernelTokenizer] = None,
) -> SampleGenerator:
    return SampleGenerator(
        x_values=load_default_coords(x_resolution=resolution, y_resolution=resolution),
        x_resolution=resolution,
        batch_size=8,
        make_captions=make_captions,
        min_expression=min_expression,
        max_expression=max_expression,
        use_fft=use_fft,
        tokenizer=tokeniser,
        iterations_per_epoch=10,
    )


def test_sampler_len():
    assert len(get_sampler()) == 10


def test_sampler_make_batch_of_samples(tokeniser):
    batch = get_sampler(tokeniser=tokeniser).make_batch_of_samples(
        kernel=gpflow.kernels.RBF()
    )
    assert batch.shape == (8, 28, 28)


def test_make_batch_of_labels__fails_with_incompatible_args():
    # single expression mode
    with pytest.raises(AssertionError):
        get_sampler(max_expression=1,
                    make_captions=False).make_batch_of_labels(expression=COMPOUND_EXPRESSION)


def test_make_batch_of_labels__compound_expression(tokeniser):
    np.testing.assert_array_equal(
        get_sampler(tokeniser=tokeniser, max_expression=4).make_batch_of_labels(
            expression=COMPOUND_EXPRESSION
        ),
        np.tile(np.array([9, 2, 6, 3, 10, 0]), (8, 1)),
    )


def test_make_batch_of_labels__single_expression():
    np.testing.assert_array_equal(
        get_sampler(max_expression=1, make_captions=False).make_batch_of_labels(
            expression=SINGLE_EXPRESSION
        ),
        np.tile(np.array([1]), (8,)),
    )


def test_make_batch__compound_expression():
    """ Attempt to make several batches and check they are the right shape.
    Here we use a tokenizer with a larger vocabulary. """

    max_products = 3  # Max length of a product kernel
    vocab = get_unique_product_kernels(max_terms=max_products)
    tokeniser = KernelTokenizer(vocabulary=vocab)
    sampler = get_sampler(resolution=16, max_expression=4, tokeniser=tokeniser)

    for _ in range(50):
        features, labels = sampler.make_batch()
        assert features.shape == (8, 16, 16)
        assert labels.shape == (8, 6)


def test_make_batch__single_expression(tokeniser):
    features, labels = get_sampler(
        resolution=16,
        make_captions=False,
    ).make_batch()
    assert features.shape == (8, 16, 16)
    assert labels.shape == (8,)
