""" Generates random samples from a compound kernel. """
from typing import List, Optional

import gpflow as gf
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from kitt.config import EPSILON
from kitt.data.sampler.utils import (
    fft_samples,
    load_random_coords,
    make_kernel_from_label,
    make_random_expression,
    make_samples,
    order_sum_kernel_by_variance,
    randomise_hyperparameters, standardise_samples,
)
from kitt.data.tokeniser import KernelTokenizer


class SampleGenerator(tf.keras.utils.Sequence):
    """ Generate samples drawn from a combination of primitive kernels. """

    def __init__(
        self,
        x_values: np.ndarray,
        x_resolution: int,
        batch_size: int,
        min_expression: int = 1,
        max_expression: int = 3,
        sigma: float = 1.0,
        use_fft: bool = False,
        add_bias: bool = False,
        standardise: bool = True,
        all_dims: bool = False,
        tokenizer: Optional[KernelTokenizer] = None,
        make_captions: bool = True,
        include_x_with_samples: bool = False,
        iterations_per_epoch: int = 10_000,
    ) -> None:
        """
        :param x_values: Coordinates for evaluating the samples
        :param x_resolution: number of coords in first dimension
        :param batch_size: Number of samples per kernel randomisation
        :param min_expression: Minimum complexity of the kernel expression
        :param max_expression: Maximum complexity of the kernel expression
        :param sigma: Standard deviation of the Gaussian prior on log hyperparameters.
        # Smaller values are easier to train, but less generalisable
        :param use_fft: whether to fft the generated samples
        :param add_bias: Whether to add a bias (constant) kernel to each subkernel
        :param standardise: force mean=0, std=1 on generated samples
        :param all_dims: Whether to activate all input dimensions (True) or a random subset (False),
        :param tokenizer: kernel expression tokenizer
        :param make_captions: Whether the labels are captions or single tokens
        :param include_x_with_samples: Boolean as to whether to include x_values alongside features
            in the batch of samples. This is necessary for regression problems where train_x values
            are not known in advance. But for data on a regular grid (eg images or most time series)
            there is no need to specify the input values explicitly.
        :param iterations_per_epoch: The number of training iterations per epoch.
        """

        self.x_values = x_values
        self.make_captions = make_captions
        self.ndims = x_values.shape[1]
        self.sigma = sigma
        self.all_dims = all_dims
        self.add_bias = add_bias
        self.use_fft = use_fft
        self.standardise = standardise
        self.batch_size = batch_size
        self.mean_function = np.zeros(x_values.shape[0])
        self.min_expression = min_expression
        self.max_expression = max_expression
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else KernelTokenizer()
        )
        if include_x_with_samples:  # x values are informative
            self.x_resolution = None
            self.y_resolution = None
        else:  # x values are at fixed grid locations
            self.x_resolution = x_resolution
            self.y_resolution = x_values.shape[0] // x_resolution
        self.iterations_per_epoch = iterations_per_epoch
        self.include_x_with_samples = include_x_with_samples

        if self.max_expression > 1:  # compound expressions require tokeniser
            assert self.tokenizer
        if not self.make_captions:
            assert self.max_expression == 1, "Longer expressions require captioning mode"
        if self.include_x_with_samples:
            assert not self.use_fft, 'FFT only supported for gridded x'

    def make_batch(self):
        """ Safe version of _make_batch, in case of cholesky error. """

        attempts = 0
        completed_sampling = False
        features, labels = None, None

        while not completed_sampling and attempts < 5:
            try:
                features, labels = self._make_batch()
            except InvalidArgumentError:
                print("Invalid kernel found - retrying with new batch")
                attempts += 1
            else:
                completed_sampling = True

        return features, labels

    def _make_batch(self):
        """
        First select random model, then randomise its hyperparameters, then draw random samples.
        """

        expression = make_random_expression(
            min_complexity=self.min_expression,
            max_complexity=self.max_expression,
            kernel_tokens=self.tokenizer.kernel_tokens,
        )
        kernel = make_kernel_from_label(expression, self.ndims, self.sigma, all_dims=self.all_dims, add_bias=self.add_bias)

        randomise_hyperparameters(kernel)
        # The randomisation process influences the expression ordering
        kernel = order_sum_kernel_by_variance(kernel)
        features = self.make_batch_of_samples(kernel)
        labels = self.make_batch_of_labels(list(expression))

        return features, labels

    def make_batch_of_labels(self, expression: List[str]):
        """ Create a suitable set of labels. """

        if self.make_captions:
            token = self.tokenizer.encode(
                    expression=expression,
                    pad=True,
                    max_complexity=self.max_expression,
                )
            label = np.tile(token, (self.batch_size, 1))

        else:
            # single expression
            assert len(expression) == 1, "Longer expressions require captioning mode"
            token = [self.tokenizer.kernel_tokens.index(expression[0])]
            # Replicate the same label across the batch
            # For SparseCategoricalCrossentropy, the expected shape of label is [batch_size]
            label = np.tile(token, (self.batch_size,))

        return label

    def make_batch_of_samples(self, kernel: gf.kernels.Kernel) -> np.ndarray:
        """ Generate samples from gpflow kernel with random hyperparameters.
        Two fundamentally different kinds of samples: one where the x values are predetermined (such as an image)
        and those where the x values are irregular (typically high dimensional problems). Only for the latter problem
        do we need to include the x values within the samples. """

        if self.include_x_with_samples:
            self.x_values = load_random_coords(n_samples=self.x_values.shape[0], n_dims=self.ndims)
            covariance = self.compute_covariance(kernel)

            samples = make_samples(covariance, self.batch_size, self.standardise)
            samples = np.expand_dims(samples, axis=-1)

            input_x = standardise_samples(self.x_values.T).T   # todo figure out best solution for normalisation
            stacked_x = np.repeat(input_x[None], self.batch_size, axis=0)

            samples = np.concatenate((stacked_x, samples), axis=-1)
        else:
            batch_shape = (self.batch_size, self.x_resolution, self.y_resolution)
            covariance = self.compute_covariance(kernel)

            samples = make_samples(covariance, self.batch_size, self.standardise)
            samples = samples.reshape(batch_shape)
            if self.use_fft:
                samples = fft_samples(samples)

        return samples

    def compute_covariance(self, kernel: gf.kernels.Kernel):
        """ For numerical stability we add a small amount of jitter, gpflow uses 1e-6 by default. """

        cov = kernel(self.x_values)
        jitter = EPSILON * np.eye(len(self.x_values))

        return cov + jitter

    def __getitem__(self, index):
        # Generate one batch of data
        return self.make_batch()

    def __len__(self) -> int:
        # Denotes the number of batches per epoch
        return self.iterations_per_epoch
