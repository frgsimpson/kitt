"""

Components helpful for assembling a transformer in TensorFlow.
Adapted from the pytorch implementation at https://github.com/PrincetonLIPS/AHGP/

"""

import tensorflow as tf

from keras.layers import Layer, LayerNormalization
from keras.layers import Dense, Dropout, Permute

from kitt.prototype.attention.scaled_dot_product_attention import MultiheadAttentionMechanism


DROPOUT_RATE = 0.1


class MHAttentionBlock(Layer):
    """
    Multiheaded Attention Layer for use in a transformer
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(MHAttentionBlock, self).__init__()

        assert num_hidden % h == 0, "Hidden dimensions not divisible by number of attention heads"

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Dense(num_hidden, use_bias=False)
        self.value = Dense(num_hidden, use_bias=False)
        self.query = Dense(num_hidden, use_bias=False)

        self.multihead = MultiheadAttentionMechanism(self.num_hidden_per_attn)

        self.permute = Permute((2, 1, 3))
        self.residual_dropout = Dropout(rate=DROPOUT_RATE)

        self.final_layer = Dense(num_hidden, activation='relu')
        self.layer_norm = LayerNormalization(epsilon=1e-6, dtype="float64")

    def call(self, key, value, query, mask=None, training=None):

        batch_size = tf.shape(key)[0]
        seq_k = tf.shape(key)[1]
        residual = value

        key = self.key(key)
        value = self.value(value)
        query = self.query(query)

        new_shape = [batch_size, seq_k, self.h, self.num_hidden_per_attn]
        key = tf.reshape(key, new_shape)
        value = tf.reshape(value, new_shape)
        query = tf.reshape(query, new_shape)

        query = self.permute(query)
        key = self.permute(key)
        value = self.permute(value)

        result, _ = self.multihead.forward(key, value, query, mask=mask, training=training)
        result = self.permute(result)
        result_shape = [batch_size, seq_k, -1]
        result = tf.reshape(result, result_shape)

        result = self.final_layer(result)

        result = self.residual_dropout(result, training=training)
        result = result + residual
        result = self.layer_norm(result)

        return result


class SelfAttentionBlock(Layer):
    """ A self attention layer. """
    def __init__(
        self,
        representation_dimension: int,
        num_attn_heads: int,
    ):
        super(SelfAttentionBlock, self).__init__()
        self.mab = MHAttentionBlock(representation_dimension, num_attn_heads)

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x, x)
