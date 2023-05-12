from kitt.config import MODEL_SAVE_DIR
from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import load_random_coords
from kitt.data.tokeniser import KernelTokenizer
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.utils.save_load_models import load_model


def get_trained_captioning_network(encoder_name: str = None, decoder_name: str = None, max_terms: int = 2):
    """ Load trained captioning network """

    # todo - implement loading of samplegenerator, currently risk mismatch of vocabulary
    sample_generator = get_default_sample_generator(max_terms=max_terms)
    encoder, _ = get_trained_classification_network(classifier_name=encoder_name, max_terms=max_terms)

    if decoder_name is None:  # Make new decoder
        rnn_units = 512
        vocab_size = sample_generator.tokenizer.n_vocabulary
        decoder = RNN_Decoder(encoder.num_hidden_units, rnn_units, vocab_size)
    else:
        filename = MODEL_SAVE_DIR / "decoders" / decoder_name
        decoder = load_model(filename)

    return encoder, decoder, sample_generator


def get_trained_classification_network(classifier_name: str, max_terms: int = 2):
    """ Load trained classification network """

    tokenizer = get_default_tokenizer(max_terms=max_terms)

    if classifier_name is None:  # Make new encoder
        classifier = ClassificationTransformer(
            num_hidden_units=128,
            num_heads=4,
            num_classes=tokenizer.n_vocabulary,
        )
    else:
        filename = MODEL_SAVE_DIR / "encoders" / classifier_name
        classifier = load_model(filename)

    return classifier, tokenizer


def get_default_sample_generator(n_dims: int = 2,
                                 resolution: int = 256,
                                 max_terms: int = 2,
                                 captions: bool = True) -> SampleGenerator:
    """ Does what it says on the tin. """

    tokenizer = get_default_tokenizer(max_terms=max_terms)
    coords = load_random_coords(n_samples=resolution, n_dims=n_dims)

    sample_generator = SampleGenerator(
        x_values=coords,
        x_resolution=resolution,
        batch_size=64,
        min_expression=1,  # Smallest caption length
        max_expression=3,  # Longest caption length
        tokenizer=tokenizer,
        make_captions=captions,
        include_x_with_samples=True,
        iterations_per_epoch=10_000
    )
    return sample_generator


def get_default_tokenizer(max_terms: int = 2) -> KernelTokenizer:
    """ Does what it says on the tin. """

    vocab = get_unique_product_kernels(max_terms=max_terms)
    return KernelTokenizer(vocabulary=vocab)
