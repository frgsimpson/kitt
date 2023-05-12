import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def accuracy_function(real, pred):
    predictions = tf.equal(real, tf.cast(tf.argmax(pred, axis=-1), real.dtype))

    padding_mask = tf.math.logical_not(tf.math.equal(real, 0))
    predictions = tf.math.logical_and(padding_mask, predictions)

    predictions = tf.cast(predictions, dtype=tf.float32)
    padding_mask = tf.cast(padding_mask, dtype=tf.float32)
    return tf.reduce_sum(predictions) / tf.reduce_sum([padding_mask])
