from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Lambda
from tensorflow.python.keras.models import Sequential


def load_encoder(embedding_dim: int, use_cnn: bool, input_shape: Tuple[int, ...]):
    if use_cnn:
        return load_convolutional_encoder(embedding_dim, 64, input_shape)

    return load_dense_encoder(embedding_dim)


def load_dense_encoder(embedding_dim: int):

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(embedding_dim, activation="relu", name="layer1"),
            tf.keras.layers.Dense(embedding_dim, activation="relu", name="layer2"),
        ]
    )


def load_convolutional_encoder(embedding_dim: int, filters: int, input_shape: Tuple[int, ...]):
    """ Shallow convolutional network. """

    channel_insertion_layer = Lambda(lambda image: tf.expand_dims(image, axis=-1))
    dim_list = list(input_shape)
    dim_list.append(1)
    image_shape = tuple(dim_list)

    return tf.keras.Sequential(
        [
            channel_insertion_layer,
            tf.keras.layers.Conv2D(filters, activation="relu", name="layer1", kernel_size=3, input_shape=image_shape),
            tf.keras.layers.Conv2D(filters, activation="relu", name="layer2", kernel_size=3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(embedding_dim, activation="relu", name="layer3"),
        ]
    )


def load_lstm_encoder(embedding_size, hidden_size=512):
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))

    model.add(TimeDistributed(Dense(embedding_size)))
    return model
