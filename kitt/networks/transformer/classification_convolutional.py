""" This network uses its translational invariance to study different datapoints across different dimensions.
 Not quite as general as permutation invariance but should still be better than a dense network to perform this task. """

import tensorflow as tf

from kitt.prototype.classifier.models import load_2d_cnn_classifier


class ClassificationCNN(tf.keras.Model):
    def __init__(
        self,
        n_classes: int,
        input_shape,  # Expected to be [batch size, dimensions, sequence, 2]
        network_identifier: str = 'shallow',
        n_channels: int = 2,  # Usually 2, for x and y values at a given (dimension, sequence) coordinate
    ):
        super(ClassificationCNN, self).__init__()

        self.cnn = load_2d_cnn_classifier(
            n_classes=n_classes,
            input_shape=input_shape,
            network_name=network_identifier,
            n_channels=n_channels,
        )

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

    def call(self, inputs):

        inputs = self.process_inputs(inputs)

        return self.cnn(inputs)
