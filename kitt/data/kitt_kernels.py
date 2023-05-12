""" Custom kernel, adapted from their default GPflow configuration. """

import gpflow as gf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Periodic(gf.kernels.Periodic):
    """
    The interface of the Periodic kernel was changed in https://github.com/GPflow/GPflow/pull/1158
    and this class provides the old version of it.
    """

    def __init__(
        self,
        lengthscales=1.0,
        variance: float = 1.0,
        period: float = 1.0,
        active_dims=None,
    ) -> None:
        base_kernel = gf.kernels.SquaredExponential(
            variance=variance,
            lengthscales=lengthscales,
            active_dims=active_dims,
        )
        super().__init__(base_kernel=base_kernel, period=period)


class Cosine(gf.kernels.Cosine):
    """
    The gpflow cosine does not permit negative frequencies, which are important for higher dimensional cases.
    Here we redefine the lengthscales Parameter with the same values, but omitting the positive transform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lengthscale_values = self.lengthscales.numpy()
        lengthscale_prior = tfp.distributions.Cauchy(np.float64(0.), np.float64(5.))
        self.lengthscales = gf.Parameter(lengthscale_values, prior=lengthscale_prior)  # Removed default positive transform


class ShiftedLinear(gf.kernels.Linear):
    """
    The standard linear kernel can only increase in variance away from the origin.
    Primarily intended to be used in product with a noise kernel.
    Larger shifts would naturally lead to large values in noise amplitude.
    To counteract this we scale the variance by the shift amplitude?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.active_dims, (list, np.ndarray, tf.Tensor)):
            ndims = len(self.active_dims)
        else:
            ndims = 1
        shift_values = np.zeros(ndims)
        self.shifts = gf.Parameter(shift_values, name='shifts', trainable=True)  # Removed default positive transform

    def K(self, X, X2=None):

        Z = self.shift_input(X)
        normalised_variance = self.variance / (1 + self.shifts ** 2)

        if X2 is None:
            return tf.matmul(Z * normalised_variance, Z, transpose_b=True)
        else:
            Z2 = self.shift_input(X2)
            return tf.tensordot(Z * normalised_variance, Z2, [[-1], [-1]])

    def K_diag(self, X):

        Z = self.shift_input(X)
        normalised_variance = self.variance / (1 + self.shifts ** 2)

        return tf.reduce_sum(tf.square(Z) * normalised_variance, axis=-1)

    def shift_input(self, X):
        return X + self.shifts


def make_linear_noise_kernel(kernel_name: str, ndims: int):
    """ Instantiate a linear noise kernel with an RBF. """

    variances = np.ones(ndims)
    lengthscales = np.ones(ndims)

    noise = gf.kernels.White()
    rbf = gf.kernels.RBF(lengthscales=lengthscales)

    if kernel_name == 'constantnoise':
        return rbf

    if kernel_name == 'linearnoise':
        linear_kernel = gf.kernels.Linear(variance=variances)
    elif kernel_name == 'offsetlinearnoise':
        linear_kernel = gf.kernels.Polynomial(variance=variances, degree=1.0, offset=0.1)
    elif kernel_name == 'shiftedlinearnoise':
        linear_kernel = ShiftedLinear(variance=variances)
    else:
        raise NotImplementedError('Unknown kernel', kernel_name)

    return noise * linear_kernel + rbf
