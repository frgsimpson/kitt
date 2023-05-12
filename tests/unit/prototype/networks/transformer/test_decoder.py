import numpy as np
import tensorflow as tf

from kitt.data.kernels import get_unique_product_kernels
from kitt.data.tokeniser import KernelTokenizer
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.transformer.transformer_decoder import (
    TransformerDecoder,
    make_look_ahead_mask,
    make_padding_mask
)


def test_permutation_invariance__decoder_only():
    """
    Ensures invariance to the ordering of the dimensions of the features provided to the decoder.
    """
    np.random.seed(1)  # Ensure test is deterministic
    vocab = get_unique_product_kernels(max_terms=3)
    tokenizer = KernelTokenizer(vocabulary=vocab)
    vocab_size = tokenizer.n_vocabulary
    decoder = TransformerDecoder(
        num_units=32,
        num_heads=4,
        num_layers=2,
        vocab_size=vocab_size
    )

    reference_features = np.random.randn(1, 6, 32)

    seq_perm_features = np.copy(reference_features)
    seq_perm_features[:, [0, 1], :] = seq_perm_features[:, [1, 0], :]

    # Ought to generate a different output when we swap an x and y value
    diff_features = np.copy(reference_features)
    diff_features[:, :, [1, 2]] = diff_features[:, :, [2, 1]]

    # Ought to generate a different output when we swap point from different sample and dimension
    switched_features = np.copy(reference_features)
    switched_features[0, 0, 1] = reference_features[0, 1, 0]
    switched_features[0, 1, 0] = reference_features[0, 0, 1]

    decoder_prompt = tf.expand_dims(tf.convert_to_tensor([16]), 0)
    # Generate masks to run the decoder without attending to future or padding values.
    look_ahead_mask = make_look_ahead_mask(tf.shape(decoder_prompt)[1])
    padding_mask = make_padding_mask(decoder_prompt)
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)

    reference_output = decoder(decoder_prompt, reference_features, False, combined_mask)
    seq_output = decoder(decoder_prompt, seq_perm_features, False, combined_mask)
    diff_output = decoder(decoder_prompt, diff_features, False, combined_mask)
    switch_output = decoder(decoder_prompt, switched_features, False, combined_mask)

    np.testing.assert_allclose(reference_output, seq_output, err_msg='Failed sequence invariance')
    assert not np.allclose(reference_output, diff_output), ' Identical outputs for swapping x and y'
    assert not np.allclose(reference_output, switch_output), ' Identical outputs for swapping off-diagonal terms.'


def test_permutation_invariance__end_to_end():
    """
    Ensures invariance to permutations of the sequence and x dimensions of the inputs to the
    original encoder carries through to the decoder in an end-to-end run.
    """
    np.random.seed(1)  # Ensure test is deterministic
    vocab = get_unique_product_kernels(max_terms=3)
    tokenizer = KernelTokenizer(vocabulary=vocab)
    vocab_size = tokenizer.n_vocabulary
    decoder = TransformerDecoder(
        num_units=32,
        num_heads=4,
        num_layers=2,
        vocab_size=vocab_size
    )
    encoder = ClassificationTransformer(
        num_hidden_units=64,
        num_heads=4,
        num_classes=vocab_size,
    )

    np.random.seed(1)  # Ensure test is deterministic

    # Set up decoder inputs
    decoder_prompt = tf.expand_dims(tf.convert_to_tensor([16]), 0)
    # Generate masks to run the decoder without attending to future or padding values.
    look_ahead_mask = make_look_ahead_mask(tf.shape(decoder_prompt)[1])
    padding_mask = make_padding_mask(decoder_prompt)
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)

    # Build encoder inputs starting from a reference input
    reference_input = np.random.rand(1, 2, 3)  # (batch_size, sequence_length, x_dimensions + 1)

    # Permute the sequence ordering (which shouldn't affect outputs)
    seq_perm_input = np.copy(reference_input)
    seq_perm_input[:, [0, 1], :] = seq_perm_input[:, [1, 0], :]

    # Permute the dimensions which again should not affect the output.
    dim_perm_input = np.copy(reference_input)
    dim_perm_input[:, :, [0, 1]] = dim_perm_input[:, :, [1, 0]]

    # Swapping x and y values ought to generate a different output.
    diff_input = np.copy(reference_input)
    diff_input[:, :, [1, 2]] = diff_input[:, :, [2, 1]]

    # Swapping points between the two x dimensions ought to generate a different output.
    switch_input = np.copy(reference_input)
    switch_input[0, 0, 1] = reference_input[0, 1, 0]
    switch_input[0, 1, 0] = reference_input[0, 0, 1]

    # Get features for each different set of inputs
    reference_features = encoder.get_representations(reference_input)
    seq_perm_features = encoder.get_representations(seq_perm_input)
    dim_perm_features = encoder.get_representations(dim_perm_input)
    diff_features = encoder.get_representations(diff_input)
    switch_features = encoder.get_representations(switch_input)

    # Pass the features through the decoder (with the same prompt and masks) to get outputs to compare.
    reference_output = decoder(decoder_prompt, reference_features, False, combined_mask)
    seq_output = decoder(decoder_prompt, seq_perm_features, False, combined_mask)
    dim_output = decoder(decoder_prompt, dim_perm_features, False, combined_mask)
    diff_output = decoder(decoder_prompt, diff_features, False, combined_mask)
    switch_output = decoder(decoder_prompt, switch_features, False, combined_mask)

    # Perform the comparisons and assert what we expect.
    np.testing.assert_allclose(reference_output, seq_output, err_msg='Failed sequence invariance')
    np.testing.assert_allclose(reference_output, dim_output, err_msg='Failed dimensional invariance')
    assert not np.allclose(reference_output, diff_output), ' Identical outputs for swapping x and y'
    assert not np.allclose(reference_output, switch_output), ' Identical outputs for swapping off-diagonal terms.'
