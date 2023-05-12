import pytest

import tensorflow as tf

from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.transformer.transformer_decoder import (
    TransformerDecoder,
    make_look_ahead_mask,
    make_padding_mask
)
from kitt.prototype.evaluate_captions import (
    get_caption_for_sample,
    multi_evaluate,
    infer_top_n_captions
)
from kitt.prototype.get_models import get_default_sample_generator


@pytest.fixture
def sample_generator() -> SampleGenerator:
    return get_default_sample_generator(max_terms=3, captions=True)


@pytest.fixture
def encoder(sample_generator: SampleGenerator):
    classifier = ClassificationTransformer(
        num_hidden_units=64,
        num_heads=4,
        num_classes=sample_generator.tokenizer.n_vocabulary
    )
    # In order to avoid building new variables within a tf.function we build the model here before
    # using it. Our model is agnostic to the input dimensions so the input shape tuple here is
    # irrelevant.
    classifier.build((1, 2, 3))
    return classifier


@pytest.fixture(scope="function")
def decoder(request, sample_generator: SampleGenerator):
    if request.param:
        return RNN_Decoder(
            embedding_dim=16,
            units=32,
            vocab_size=sample_generator.tokenizer.n_vocabulary
        )
    else:
        decoder = TransformerDecoder(
            num_units=32,
            num_heads=4,
            num_layers=2,
            vocab_size=sample_generator.tokenizer.n_vocabulary
        )
        # In order to avoid building new variables within a tf.function we build the model here
        # before using it. Here we build it by running a forward pass.
        mock_input = tf.constant([[1]])
        look_ahead_mask = make_look_ahead_mask(tf.shape(mock_input)[1])
        padding_mask = make_padding_mask(mock_input)
        combined_mask = tf.maximum(padding_mask, look_ahead_mask)
        mock_features = tf.random.normal((1, 2, 64))
        decoder(tf.constant([[1]]), mock_features, False, combined_mask)
        return decoder


@pytest.mark.parametrize("decoder", [True, False], indirect=["decoder"])
def test_get_caption_for_sample(decoder, encoder, sample_generator):
    samples, _ = sample_generator.make_batch()
    caption, caption_probability = get_caption_for_sample(
        samples[0],
        sample_generator.tokenizer,
        sample_generator.max_expression,
        encoder,
        decoder
    )
    assert isinstance(caption, list)
    assert caption and len(caption) <= sample_generator.max_expression
    assert all(isinstance(elem, str) for elem in caption)
    assert 0 < caption_probability < 1


@pytest.mark.parametrize("decoder", [True, False], indirect=["decoder"])
def test_multi_evaluate(decoder, encoder, sample_generator):
    samples, _ = sample_generator.make_batch()
    captions = multi_evaluate(
        samples[0], sample_generator, decoder, encoder, n_evals=3
    )
    assert isinstance(captions, list)
    assert all(isinstance(elem, tuple) for elem in captions)
    assert all(
        caption and len(caption) <= sample_generator.max_expression and 0 < p_caption < 1
        for caption, p_caption in captions
    )
    probs = [c[1] for c in captions]
    assert probs == sorted(probs, reverse=True), \
        "Failed. Captions not in descending order of likelihood."


@pytest.mark.parametrize("decoder", [False, True], indirect=["decoder"])
def test_infer_top_n_captions(decoder, encoder, sample_generator):
    samples, _ = sample_generator.make_batch()
    captions, weights = infer_top_n_captions(
        encoder, decoder, sample_generator, samples[0, :, :-1], samples[0, :, -1:], n_captions=4
    )
    assert isinstance(captions, tuple)
    assert all(isinstance(caption, list) for caption in captions)
    assert all(isinstance(elem, str) for caption in captions for elem in caption)
    assert all(
        caption and len(caption) <= sample_generator.max_expression and 0 < p_caption < 1
        for caption, p_caption in zip(captions, weights)
    )
    assert list(weights) == sorted(weights, reverse=True), \
        "Failed. Captions not in descending order of likelihood."
