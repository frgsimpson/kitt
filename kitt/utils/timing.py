import time
from typing import Callable, Tuple

import tensorflow as tf
import numpy as np


@tf.function
def time_inference(
        inference_fn: Callable[[tf.Tensor], tf.Tensor],
        x_dims: int,
        seq_length: int,
        repeats: int
) -> Tuple[float, float]:
    data = tf.random.normal((1, seq_length, x_dims + 1))
    # Run forward pass to ensure everything is initialised
    inference_fn(data)
    times = []
    for _ in range(repeats):
        start = time.time()
        inference_fn(data)
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)
