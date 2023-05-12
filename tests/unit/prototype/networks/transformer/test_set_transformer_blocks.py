""" Blocks should be permutation equivariant. """

import numpy as np

from kitt.networks.transformer.transformer_blocks import MHAttentionBlock
from kitt.networks.transformer.set_transformer_blocks import MultiHeadAttentionBlock, MLP


def test_mha_equivariance():
    """ Check that a shifted input in the block leads to a shifted output. """
    d = 2
    num_hidden_units = 8

    reference_input = np.random.rand(1, d, num_hidden_units)  # (batch_size, sequence_length, x_dimensions + 1)
    shifted_input = np.roll(reference_input, 1, axis=1)

    mha_block = MultiHeadAttentionBlock(d=num_hidden_units, h=4, rff=MLP(num_hidden_units))

    reference_output = mha_block(reference_input, reference_input)
    shifted_output = mha_block(shifted_input, shifted_input)
    shifted_output = np.roll(shifted_output, -1, axis=1)

    np.testing.assert_allclose(reference_output, shifted_output)


def test_mhattention_equivariance():
    """ Check that a shifted input in the MHAttentionBlock leads to a shifted output. """

    d = 2
    num_hidden_units = 8

    reference_input = np.random.rand(1, d, num_hidden_units)  # (batch_size, sequence_length, x_dimensions + 1)
    shifted_input = np.roll(reference_input, 1, axis=1)

    mha_block = MHAttentionBlock(num_hidden_units, 4)

    reference_output = mha_block(reference_input, reference_input, reference_input)
    shifted_output = mha_block(shifted_input, shifted_input, shifted_input)
    shifted_output = np.roll(shifted_output, -1, axis=1)

    np.testing.assert_allclose(reference_output, shifted_output)

