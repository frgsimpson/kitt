""" Check that the transformer is permutation invar"""
import numpy as np
import pytest

from kitt.networks.transformer.classification_transformer import ClassificationTransformer


@pytest.mark.parametrize("use_set", [True, False])
def test_permutation_invariance(use_set: bool):

    n_classes = 3
    classifier = ClassificationTransformer(100, 4, n_classes, use_set=use_set)

    np.random.seed(1)  # Ensure test is deterministic
    reference_input = np.random.rand(1, 2, 3)  # (batch_size, sequence_length, x_dimensions + 1)
    seq_perm_input = np.copy(reference_input)
    seq_perm_input[:, [0, 1], :] = seq_perm_input[:, [1, 0], :]
    dim_perm_input = np.copy(reference_input)
    dim_perm_input[:, :, [0, 1]] = dim_perm_input[:, :, [1, 0]]

    # Ought to generate a different output when we swap an x and y value
    diff_input = np.copy(reference_input)
    diff_input[:, :, [1, 2]] = diff_input[:, :, [2, 1]]

    # Ought to generate a different output when we swap point from different sample and dimension
    switch_input = np.copy(reference_input)
    switch_input[0, 0, 1] = reference_input[0, 1, 0]
    switch_input[0, 1, 0] = reference_input[0, 0, 1]

    reference_output = classifier(reference_input)
    seq_output = classifier(seq_perm_input)
    dim_output = classifier(dim_perm_input)
    diff_output = classifier(diff_input)
    switch_output = classifier(switch_input)

    ref_eq_output = classifier.encode_sequence(reference_input)
    alt_seq_output = classifier.encode_sequence(seq_perm_input)
    np.testing.assert_allclose(ref_eq_output, alt_seq_output, err_msg='Encoder failed sequence invariance')

    np.testing.assert_allclose(reference_output, seq_output, err_msg='Failed sequence invariance')
    np.testing.assert_allclose(reference_output, dim_output, err_msg='Failed dimensional invariance')
    assert not np.allclose(reference_output, diff_output), ' Identical outputs for swapping x and y'
    assert not np.allclose(reference_output, switch_output), ' Identical outputs for swapping off-diagonal terms.'

    assert reference_output.numpy().shape == (1, n_classes), 'Unexpected output shape from classifier'



