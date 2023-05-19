import tensorflow as tf
from tensorflow.keras.layers import Dense, Permute
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Lambda

from kitt.networks.transformer.transformer_blocks import SelfAttentionBlock
from kitt.networks.transformer.set_transformer_blocks import SetAttentionBlock, MLP

tf.keras.backend.set_floatx("float64")
DEFAULT_N_ENCODER_BLOCKS = 6


def soft_transform(x):
    return tf.nn.log_softmax(x, axis=2)


def pool_dimensions(x):  # todo compare performance with reduce_mean in place of reduce_logsumexp
    return tf.math.reduce_logsumexp(x, axis=1, keepdims=False)


def pool_sequence(x, axis: int = 1, keepdims: bool = True):
    return tf.math.reduce_mean(x, axis=axis, keepdims=keepdims)


class ClassificationTransformer(tf.keras.Model):
    def __init__(
        self,
        num_hidden_units: int,
        num_heads: int,
        num_classes: int,
        n_encoder_blocks: int = DEFAULT_N_ENCODER_BLOCKS,
        n_dim_encoder_blocks: int = DEFAULT_N_ENCODER_BLOCKS,
        use_set: bool = False,  # Which type of attention block to use
    ):
        super().__init__()

        self.num_hidden_units = num_hidden_units
        self.num_heads = num_heads
        self.num_classes = num_classes

        def make_block():
            if use_set:
                block = SetAttentionBlock(
                            representation_dimension=num_hidden_units,
                            num_attn_heads=num_heads,
                            feed_fwd_encoder=MLP(num_hidden_units),
                        )
            else:
                block = SelfAttentionBlock(num_hidden_units, num_heads)

            return block

        self.transpose_for_processing = Permute((2, 1, 3))
        self.simple_encoder = Dense(num_hidden_units, activation=None, use_bias=False)

        seq_block_list = []
        for i in range(n_encoder_blocks):
            block = make_block()
            seq_block_list.append(block)

        dim_block_list = []
        for i in range(n_dim_encoder_blocks):
            block = make_block()
            dim_block_list.append(block)

        self.seq_encoding_transformer = Sequential(seq_block_list)
        self.encode_dimensions = Sequential(dim_block_list)

        self.sequence_pooling_layer = Lambda(pool_sequence)
        self.dense_layers = Sequential([
                Dense(num_hidden_units, activation="relu"),
                Dense(num_hidden_units, activation="relu"),
            ])

        self.classification_layers = Sequential([
            Dense(num_classes, activation=None),
            Lambda(soft_transform),
            Lambda(pool_dimensions),
        ])

    @staticmethod
    def process_inputs(inputs):
        """
        Transform inputs to split out the x dimensions for dimension-agnostic processing.
        Input shape (batch_size, sequence_length, x_dimensions + 1)
        Output shape (batch_size, sequence_length, x_dimensions, 2)
        """
        x, y = inputs[..., :-1], inputs[..., -1:]
        num_x_dims = tf.shape(x)[-1]
        x = tf.expand_dims(x, axis=-1)
        y = tf.repeat(tf.expand_dims(y, axis=-1), num_x_dims, axis=2)
        return tf.concat([x, y], axis=-1)

    def encode_sequence(self, inputs, training=None):

        # Initial shape is (batch_size, sequence_length, x_dimensionality + 1)
        inputs = self.process_inputs(inputs)
        # Now (batch_size, sequence_length, x_dimensionality, 2)

        batch_size = tf.shape(inputs)[0]
        num_x_dims = tf.shape(inputs)[2]

        # This is then transposed to (batch_size, x_dimensionality, sequence_length, 2)
        preprocessed_x = self.transpose_for_processing(inputs)

        # Encoding with a vanilla MLP yields shape
        # (batch_size, x_dimensionality, sequence_length, num_hidden_units)
        transformed_values = self.simple_encoder(preprocessed_x)

        # Absorb the x_dimensions into the batch size for easier parallel processing
        # Shape becomes (batch_size * x_dimensionality, sequence_length, num_hidden_units)
        x_absorb_x_dims_into_batch = tf.reshape(
            transformed_values, (batch_size * num_x_dims, -1, self.num_hidden_units)
        )
        # The initial attention encoding doesn't change the shape.
        # Shape is still (batch_size * x_dimensionality, sequence_length, num_hidden_units)
        attention_encoded_values = self.seq_encoding_transformer(
            x_absorb_x_dims_into_batch, training=training
        )

        # Pooling over sequence drops the sequence dimension down to 1
        # Shape becomes (batch_size * x_dimensionality, 1, num_hidden_units)
        per_dimension_representations = self.sequence_pooling_layer(attention_encoded_values)

        # We can now reconstruct the x_dimensions
        # Shape becomes (batch_size, x_dimensionality, num_hidden_units)
        return tf.reshape(
            per_dimension_representations, (-1, num_x_dims, self.num_hidden_units)
        )

    @tf.function
    def get_representations(self, x, include_dense: bool = True, training=None):
        """ A complete forward pass, except for final pooling over dimensions.
         There is also an option to skip the final dense layers"""

        x = self.encode_sequence(x, training=training)
        x = self.encode_dimensions(x, training=training)

        if include_dense:
            x = self.dense_layers(x)

        return x

    def call(self, x, training=None):
        # The generation of final representations is split out to enable easy access to
        # representations once the classification pre-training is complete and the model saved.
        x = self.get_representations(x, training=training)

        # Pool over dimensions to leave classification logits of shape (batch_size, num_classes)
        return self.classification_layers(x)


