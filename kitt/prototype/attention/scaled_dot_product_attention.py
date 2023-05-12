# Copied from https://github.com/arrigonialberto86/set_transformer/blob/master/set_transformer/layers/attention.py

"""

Multi-headed attention to be used until the tensorflow implementation is in the main tf build
https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention

Copied from the set transformer work for initial consistency with the set transformer blocks we use.

"""

import tensorflow as tf


# https://www.tensorflow.org/tutorials/text/transformer, appears in "Attention is all you need" NIPS 2018 paper
from gpflow import default_float
from tensorflow.python.keras.layers import Dropout


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float64)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor before softmax to ensure zeros at masked positions.
    # The mask is used to conceal elements in the query dimension.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model, autocast=False)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


class MultiheadAttentionMechanism(tf.Module):
    """
    Multihead attention mechanism (dot attention)
    Adapted from pytorch version https://github.com/PrincetonLIPS/AHGP/blob/4fe60d861410047a9e60823b6acc7f9c30f2ddf4/ahgp/nn/module.py#L27
    """

    def __init__(self, num_hidden_k, dropout_p=0.1):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttentionMechanism, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = Dropout(rate=dropout_p)

    def forward(self, key, value, query, mask=None, training=None):
        # Get attention score
        # query, key, value: B x h x N x dv
        key_transpose = tf.transpose(key, (0, 1, 3, 2))
        attn = tf.matmul(query, key_transpose)
        # attn = tf.matmul(query, key.transpose(2, 3))  # B x h x N x N
        attn = attn / tf.math.sqrt(tf.cast(self.num_hidden_k, default_float()))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = tf.math.softmax(attn, axis=-1)
        # Dropout
        attn = self.attn_dropout(attn, training=training)
        # Get Context Vector
        result = tf.matmul(attn, value)

        return result, attn
