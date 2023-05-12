# Referencing https://arxiv.org/pdf/1810.00825.pdf
# and the original PyTorch implementation https://github.com/TropComplique/set-transformer/blob/master/blocks.py

"""

Set transformer in TensorFlow.
Taken from https://github.com/arrigonialberto86/set_transformer

"""

import tensorflow as tf
from tensorflow import repeat
from keras.layers import Layer, LayerNormalization
from tensorflow.python.keras.layers import Dense

from kitt.prototype.attention.scaled_dot_product_attention import MultiHeadAttention


class MLP(Layer):

    def __init__(self, num_hidden_units):
        super().__init__()

        self.l1 = Dense(num_hidden_units, activation="relu")
        self.l2 = Dense(num_hidden_units, activation="relu")
        self.l3 = Dense(num_hidden_units, activation=None)

    def call(self, x):
        return self.l3(self.l2(self.l1(x)))


class MultiHeadAttentionBlock(Layer):
    def __init__(self, d: int, h: int, rff: Layer):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead = MultiHeadAttention(d, h)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6, dtype="float64")
        self.layer_norm2 = LayerNormalization(epsilon=1e-6, dtype="float64")
        self.rff = rff

    def call(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SetAttentionBlock(Layer):
    def __init__(
        self,
        representation_dimension: int,
        num_attn_heads: int,
        feed_fwd_encoder: Layer,
    ):
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(
            representation_dimension, num_attn_heads, feed_fwd_encoder
        )

    def call(self, x, training=None):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(Layer):
    def __init__(self, d: int, m: int, h: int, rff1: Layer, rff2: Layer):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            rff1, rff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super(InducedSetAttentionBlock, self).__init__()
        self.mab1 = MultiHeadAttentionBlock(d, h, rff1)
        self.mab2 = MultiHeadAttentionBlock(d, h, rff2)
        self.inducing_points = tf.random.normal(shape=(1, m, d))

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = tf.shape(x)[0]
        p = self.inducing_points
        p = repeat(p, b, axis=0)  # shape [b, m, d]

        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiHeadAttention(Layer):
    def __init__(
        self,
        input_dim: int,
        num_seed_vectors: int,
        num_heads: int,
        feed_fwd_encoder: Layer,
        pooling_feed_fwd_encoder: Layer,
    ):
        """
        Arguments:
            input_dim: an integer, input dimension.
            num_seed_vectors: an integer, number of seed vectors.
            num_heads: an integer, number of heads.
            feed_fwd_encoder: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
            feed_fwd_encoder: a module used on inputs to the pooling attention block,
                consisting of row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and returns a float tensor
                with the same shape.
        """
        super(PoolingMultiHeadAttention, self).__init__()
        self.mab = MultiHeadAttentionBlock(input_dim, num_heads, feed_fwd_encoder)
        self.seed_vectors = tf.Variable(
            tf.random.normal(shape=(1, num_seed_vectors, input_dim), dtype=tf.float64)
            / input_dim,
            trainable=True,
            name="SeedVectors",
        )
        self.rff_s = pooling_feed_fwd_encoder

    @tf.function
    def call(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d]
        """
        b = tf.shape(z)[0]
        s = self.seed_vectors
        s = repeat(s, b, axis=0)  # shape [b, k, d]
        return self.mab(s, self.rff_s(z))
