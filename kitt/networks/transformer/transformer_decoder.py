"""
Code for a transformer-based decoder for KITT.
Code inspired by https://www.tensorflow.org/tutorials/text/transformer
"""

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Lambda

from kitt.data.tokeniser import PAD_INDEX
from kitt.networks.transformer.set_transformer_blocks import MLP
from kitt.prototype.attention.scaled_dot_product_attention import MultiHeadAttention


def soft_transform(x):
    return tf.nn.log_softmax(x, axis=2)


def pool_dimensions(x):
    return tf.math.reduce_mean(x, axis=1, keepdims=False)


def make_look_ahead_mask(seq_length: int) -> tf.Tensor:
    """
    Makes a mask for training the decoder.
    The mask is of size seq_length x seq_length as we reveal one additional token in each row.
    This then acts to train on each subset of the target sequence one by one.

    :param seq_length: The length of the sequence to be trained on.
    :param batch_size: The number of items in the batch to be passed through the decoder transformer
    :return: A tensor of size (batch_size, 1, seq_length, seq_length) which can be used to mask
        future information to avoid attending to future values.
    """
    # This yields a matrix where values above the leading diagonal are 1 (denoting masked tokens)
    # and zero otherwise.
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length), dtype=tf.float64), -1, 0)
    return mask


def make_padding_mask(sequence: tf.Tensor) -> tf.Tensor:
    """
    Make a mask to cover padding so that it is not attended to (while other content can be).

    :param sequence: The (batch of) sequence(s) to generate a mask for.
    :return: A tensor of size (batch_size, 1, sequence_length, 1) used to mask padding entries in
        the provided sequence so that they are not attended to.
    """
    mask = tf.cast(tf.math.equal(sequence, PAD_INDEX), tf.float32)

    # Add extra dimensions to make shape compatible with attention logits.
    return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float64)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, num_heads, p_dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.target_sequence_attention = MultiHeadAttention(n_hidden, num_heads)
        self.feature_to_output_attention = MultiHeadAttention(n_hidden, num_heads)

        self.mlp = MLP(n_hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(p_dropout)
        self.dropout2 = tf.keras.layers.Dropout(p_dropout)
        self.dropout3 = tf.keras.layers.Dropout(p_dropout)

    def call(self, target, features, training, look_ahead_mask, padding_mask=None):
        # target shape: (batch_size, target_seq_len, target_dim)
        # features shape: (batch_size, original_data_x_dims, features_dim)

        # Part 1: Introspection on the target sequence
        attn1 = self.target_sequence_attention(
            q=target, k=target, v=target, mask=look_ahead_mask
        )  # -> (batch_size, target_seq_len, n_hidden)
        attn1 = self.dropout1(attn1, training=training)
        # Include residual link before applying layer normalisation.
        out1 = self.layernorm1(attn1 + target)

        # Part 2: Process target sequence by paying attention to the features
        attn2 = self.feature_to_output_attention(
            q=out1, k=features, v=features, mask=padding_mask
        )  # -> (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # Include residual link before applying layer normalisation.
        out2 = self.layernorm2(attn2 + out1)

        # Part 3: Generate output with simple forward pass.
        ffn_output = self.mlp(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # Include residual link before applying layer normalisation.
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


class TransformerDecoder(tf.keras.Model):
    def __init__(
            self,
            num_units: int,
            num_heads: int,
            num_layers: int,
            vocab_size: int,
            p_dropout: float = 0.1
    ):
        super(TransformerDecoder, self).__init__()

        self.num_units = num_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.encode_target_sequence = tf.keras.layers.Embedding(vocab_size, num_units)
        self.dec_layers = [DecoderLayer(num_units, num_heads, p_dropout)
                           for _ in range(num_layers)]
        self.p_dropout = p_dropout
        self.dropout = tf.keras.layers.Dropout(p_dropout)

        self.pool_dimensions = Sequential([
            # Lambda(soft_transform),
            Lambda(pool_dimensions),
        ])

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, target_seq, features, training, look_ahead_mask, padding_mask=None):

        # Initially receive separate features from each dimension
        features = self.pool_dimensions(features)

        encoded_target_seq = self.encode_target_sequence(target_seq)
        x = self.dropout(encoded_target_seq, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, features, training, look_ahead_mask, padding_mask)

        # x shape: (batch_size, target_seq_len, d_model)
        # output shape: (batch_size, target_seq_len, vocab_size)
        return self.final_layer(x)

    def get_logits_for_prompt(self, prompt, features):
        return get_logits_from_transformer_decoder(self, prompt, features)[:, -1, :]


def get_logits_from_transformer_decoder(
        decoder: TransformerDecoder,
        prompt: tf.Tensor,
        features: tf.Tensor
) -> tf.Tensor:
    look_ahead_mask = make_look_ahead_mask(tf.shape(prompt)[1])
    padding_mask = make_padding_mask(prompt)
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)
    logits = decoder(prompt, features, False, combined_mask)
    return logits
