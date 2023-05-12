""" Tools for performing inference with a classifier. """

import numpy as np
import tensorflow as tf


def infer_top_n_expressions(model, tokenizer, train_x, train_y, n_expressions=1, kernel_only_vocab=True):
    """
    Identify the most likely kernel expressions given a keras model and a training set.
    """

    input_data = np.concatenate([train_x, train_y], axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    indices, weights = infer_top_n_indices(model, input_data, n_expressions)

    expression_list = []
    for index in indices:
        if kernel_only_vocab:
            kernel_name = tokenizer.kernel_tokens[index]
        else:
            # Accounts for pad token at the start of the vocab and start and end tokens at the end.
            kernel_name = tokenizer.index_word[index]
        expression = [kernel_name]  # todo only valid for classifier, need to treat captions differently
        expression_list.append(expression)

    return expression_list, weights.tolist()


def infer_top_n_indices(model, input_data, top_n: int):
    """
    Evaluated N top predictions of a keras model
    model: A pretrained Keras model
    input_data: 'Training' data from the experiment, will also be used to train the GP
    top_n: The number of kernels to identify
    """

    logits = model.predict(input_data, batch_size=1).flatten()
    soft_prob = tf.nn.softmax(logits).numpy()

    # Extract top n classes
    sorted_arguments = (-soft_prob).argsort()
    return sorted_arguments[:top_n], soft_prob[sorted_arguments][:top_n]
