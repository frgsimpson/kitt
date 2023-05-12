import json
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union, Dict, List

from kitt.data.kernels import get_unique_product_kernels
from kitt.data.tokeniser import KernelTokenizer
from kitt.prototype.kitt_end_to_end import KITT
from kitt.networks.transformer.classification_convolutional import ClassificationCNN
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.prototype.classifier.models import load_2d_cnn_classifier, load_dense_classifier
from kitt.networks.transformer.transformer_decoder import (
    TransformerDecoder,
    make_look_ahead_mask,
    make_padding_mask
)


def instantiate_model(
        network_identifier: str,
        n_classes: int,
        batch_size: int = 64,
        hidden_units: int = 128,
        attn_heads: int = 4,
        num_input_dims: int = 4,
        include_x_with_samples: bool = False,
        resolution: int = 64,
        sample_shape: Tuple[int, int, int] = (64, 64, 3),
        rnn_units: int = 512,
        num_layers: int = 3,
        vocab_size: int = 97,
        representation_dimensionality: int = 64,
        **kwargs
):
    """
    Instantiates a classification model (including the decoder) given the arguments needed in construction.
    """
    # TODO tidy this up as it's a little ugly
    if network_identifier == "classifier-dense":
        model = load_dense_classifier(n_classes=n_classes)
    elif network_identifier == "classifier-transformer":
        model = ClassificationTransformer(
            num_hidden_units=hidden_units,
            num_heads=attn_heads,
            num_classes=n_classes,
        )
        # In order to load the weights later we need to first build the model. Our model is agnostic
        # to the input dimensions so the input shape tuple here is irrelevant.
        model.build((1, 2, 3))
    elif network_identifier == "decoder-rnn":
        model = RNN_Decoder(hidden_units, rnn_units, n_classes)
    elif network_identifier == "decoder-transformer":
        model = TransformerDecoder(
            hidden_units,
            attn_heads,
            num_layers,
            vocab_size,
            kwargs.get("p_dropout", 0.1)
        )
        mock_input = tf.constant([[1]])
        look_ahead_mask = make_look_ahead_mask(tf.shape(mock_input)[1])
        padding_mask = make_padding_mask(mock_input)
        combined_mask = tf.maximum(padding_mask, look_ahead_mask)
        mock_features = tf.random.normal((1, 2, representation_dimensionality))
        model(tf.constant([[1]]), mock_features, False, combined_mask)
    else:
        if include_x_with_samples:
            input_shape = (batch_size, resolution, num_input_dims, 2)
            model = ClassificationCNN(
                n_classes=n_classes,
                input_shape=input_shape,  # Expected to be [batch size, sequence, dimensions, 2]
                network_identifier=network_identifier,
            )
        else:
            model = load_2d_cnn_classifier(
                n_classes=n_classes,
                input_shape=sample_shape,
                network_name=network_identifier,
                n_channels=1,
            )
    return model


def save_model(
        model: tf.keras.models.Model,
        model_save_dir: Path,
        model_construction_args: Dict[str, Union[str, int, Tuple, List, float]]
):
    """
    Save a model to disk.
    Note that unfortunately saved models may not work when loaded if the codebase has changed
    significantly between the save and load events.


    :param model: The keras model to be saved
    :param model_save_dir: The directory in which to save the model weights and details.
    :param model_construction_args: The arguments used to instantiate the model. Saved to avoid
        having to remember or work out the parameters used to instantiate the model later.
    """
    # Save entire model weights and the arguments used to construct it to facilitate easy
    # loading.
    print(f"Saving model weights to {model_save_dir}")
    model.save_weights(str(model_save_dir / "saved_model"))
    with open(str(model_save_dir / "model_construction_args.json"), "w") as json_file:
        json.dump(model_construction_args, json_file)


def load_model(
        model_path_dir: Path,
):
    """
    Instantiates a model and loads previously saved weights into it.

    :param model_path_dir: The path to the folder containing the model weights and the arguments
        needed to reconstruct the model.
    """
    with open(str(model_path_dir / "model_construction_args.json"), "r") as construction_args_file:
        model_construction_kwargs = json.load(construction_args_file)
    model = instantiate_model(**model_construction_kwargs)
    model.load_weights(str(model_path_dir / "saved_model"))
    return model

def load_classifier_transformer(
        base_path: Path,
        encoder_identifier: str,
        max_products: int,
        max_expression_length: int,
        min_expression_length: int = 1
) -> ClassificationTransformer:
    encoder = load_model(base_path / "encoders" / encoder_identifier)
    tokenizer = KernelTokenizer(get_unique_product_kernels(max_terms=max_products))
    return encoder

def load_kitt(
        base_path: Path,
        encoder_identifier: str,
        decoder_identifier: str,
        max_products: int,
        max_expression_length: int,
        min_expression_length: int = 1
) -> KITT:
    encoder = load_model(base_path / "encoders" / encoder_identifier)
    decoder = load_model(base_path / "decoders" / decoder_identifier)
    tokenizer = KernelTokenizer(get_unique_product_kernels(max_terms=max_products))
    return KITT(encoder, decoder, tokenizer, max_expression_length, min_expression_length)
