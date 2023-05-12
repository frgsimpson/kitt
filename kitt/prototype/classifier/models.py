from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Lambda

from kitt.prototype.classifier.make_resnext import get_model

DEFAULT_N_LAYERS = 2
DEFAULT_LAYER_WIDTH = 128


def load_dense_classifier(
    n_classes: int,
    n_layers: int = DEFAULT_N_LAYERS,
    layer_width: int = DEFAULT_LAYER_WIDTH,
):

    input_layer = keras.layers.Flatten()
    output_layer = keras.layers.Dense(n_classes)

    layers = [input_layer]
    for _ in range(n_layers):
        layers.append(keras.layers.Dense(layer_width, activation="relu"))
    layers.append(output_layer)

    return keras.Sequential(layers)


def load_2d_cnn_classifier(n_classes: int, input_shape, network_name: str, n_channels: int = 1):
    """
    Load a CNN model. Network_name can be eg'resnext38_32x4d' or "resnext14_16x4d".
    Or pass None for a default shallow network
    """

    in_size = (input_shape[0], input_shape[1])
    if network_name == "shallow":
        model = load_shallow_cnn(n_classes=n_classes, input_shape=input_shape)
    else:
        model = get_model(
            network_name,
            pretrained=False,
            in_channels=n_channels,
            in_size=in_size,
            classes=n_classes,
        )

    return model


def load_shallow_cnn(n_classes: int, input_shape: Tuple[int]):
    """ Create a simple keras CNN. """

    dim_list = list(input_shape)
    if len(dim_list) == 3:  # Add channel by hand
        dim_list.append(1)
        channel_insertion_layer = Lambda(lambda image: tf.expand_dims(image, axis=-1))
    else:
        channel_insertion_layer = Lambda(lambda image: image)

    image_shape = tuple(dim_list)

    model = tf.keras.models.Sequential([
        channel_insertion_layer,
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=image_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_classes)
    ])
    return model
