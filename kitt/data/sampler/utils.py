""" Tools to assist the samplers. """
import re
from typing import List, Optional, Sequence

import gpflow as gf
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.kernels import Kernel, Product, Sum

from kitt.data.kernels import Periodic, choose_active_dims, get_kernel_map
from kitt.data.kitt_kernels import ShiftedLinear
from kitt.data.priors import set_default_priors_on_hyperparameters

MULTIPLICATION_TOKEN = "*"
BANDWIDTH_KERNELS = (gf.kernels.IsotropicStationary, Periodic)


def randomise_hyperparameters(module: gf.base.Module) -> None:
    """Sets hyperparameters to random samples from their prior distribution.
    :param module: Any GPflow module, such as a Kernel or GPModel
    """
    for parameter in module.trainable_parameters:
        if parameter.prior is not None:
            sample_shape = parameter.shape

            random_sample = parameter.prior.sample(sample_shape)
            parameter.assign(random_sample)


def load_default_coords(
    x_resolution: int, y_resolution: int, max_length: float = 5
) -> np.ndarray:
    """ Creates 2d input coordinates on a grid centred on zero. """

    min_x = - 0.5 * max_length
    max_x = 0.5 * max_length
    Xbase, Ybase = np.meshgrid(
        np.linspace(min_x, max_x, x_resolution), np.linspace(min_x, max_x, y_resolution)
    )
    return np.concatenate((Xbase.flatten()[:, None], Ybase.flatten()[:, None]), axis=1)


def load_random_coords(n_samples: int, n_dims: int, max_length: float = 5) -> np.ndarray:
    """ Set the x coordinates on which we'll draw samples for higher dimensional tasks.
    Most kernels dont care about shifting coords but the Linear kernel is sensitive
     so here we ensure they are centred at zero. """

    return np.random.rand(n_samples, n_dims) * max_length - max_length / 2


def fft_samples(samples, use_mag=True) -> np.ndarray:
    """Fourier transform the samples and split into real and imaginary components.
    Note that we need only keep half of the fft."""

    samples = np.fft.fft2(samples)
    nyq_index = samples.shape[-1] // 2  # + 1
    if use_mag:
        args = np.abs(samples)[:, :, :nyq_index]
        angles = np.angle(samples)[:, :, :nyq_index]

        samples = np.concatenate((args, angles), axis=-1)  # todo for cnn assign channels
    else:

        real = np.real(samples)[:, :, :nyq_index]
        imag = np.imag(samples)[:, :, :nyq_index]
        samples = np.concatenate((real, imag), axis=-1)  # todo for cnn assign channels

    return samples


def make_samples(covariance, n_samples: int, standardise: bool = True) -> np.ndarray:

    tf_covariance = tf.constant(covariance, dtype=tf.float64)
    tf_samples = tf_sampling(tf_covariance, n_samples)

    samples = tf_samples.numpy()
    if standardise:
        samples = standardise_samples(samples)

    return samples


def standardise_samples(samples):

    samples = np.atleast_2d(samples)

    mean = np.mean(samples, axis=1)[:, None]
    std = np.std(samples, axis=1)[:, None]
    return (samples - mean) / std


@tf.function
def tf_sampling(covariance, n_samples: int):
    L = tf.linalg.cholesky(covariance)
    v = tf.random.normal([1, covariance.shape[1], n_samples], dtype=tf.float64)
    samples = L @ v
    return tf.squeeze(tf.transpose(samples))


def order_sum_kernel_by_variance(kernel: Kernel) -> Kernel:
    """ Reassign variances in a Sum such that they are in descending order. """

    if isinstance(kernel, Sum):
        variance_values = extract_variances(kernel.kernels)
        variance_values.sort(reverse=True)
        assign_variances(kernel.kernels, variance_values)

    return kernel


def extract_variances(kernels: List):
    """ Find variances for a list of kernels. Assumes any product kernels have
    variance stored in final term, with the other terms being unity. """

    variance_values = []
    for k in kernels:
        var = get_kernel_variance(k)
        variance_values.append(var)

    return variance_values


def set_variance(kernel: Kernel, variance) -> Kernel:
    """
    Assign a particular variance to a kernel.
    Product kernels are constructed such that subkernels have untrainable unit variance,
    except for the final one in the list. For linear kernels we average over input dimensions.
    """

    if isinstance(kernel, Product):
        set_variance(kernel.kernels[-1], variance)
    elif isinstance(kernel, Periodic):
        set_variance(kernel.base_kernel, variance)
    elif isinstance(kernel, gf.kernels.Linear):
        current_variance = tf.reduce_mean(kernel.variance).numpy()
        scale_factor = variance / current_variance
        kernel.variance *= scale_factor
    else:
        kernel.variance.assign(variance)

    return kernel


def assign_variances(kernels: List, variances: Sequence):

    for k, v in zip(kernels, variances):
        set_variance(k, v)


def get_kernel_variance(kernel: Kernel) -> float:
    """
    Returns the total variance associated with a kernel.
    If this is a Linear kernel, or contains a linear kernel, the variance is averaged
    over input dimensions
    """

    if isinstance(kernel, gf.kernels.Linear):
        variance = tf.reduce_mean(kernel.variance).numpy()
    elif hasattr(kernel, "variance"):
        variance = kernel.variance.numpy()
    elif hasattr(kernel, "base_kernel"):
        variance = get_kernel_variance(kernel.base_kernel)
    elif isinstance(kernel, Sum):
        variance = np.sum([get_kernel_variance(k) for k in kernel.kernels])
    elif isinstance(kernel, Product):
        variance = np.product([get_kernel_variance(k) for k in kernel.kernels])
    else:
        raise NotImplementedError("Unsupported kernel found", kernel)

    return variance


def make_random_expression(
    min_complexity: int,
    max_complexity: int,
    kernel_tokens: Optional[List[str]] = None,
    weight_simple_expressions: bool = False,
) -> Sequence[str]:
    """
    Create our chosen combination of kernels. For now assume only the addition operator.

    :param min_complexity: minimum complexity of the expression
    :param max_complexity: maximum complexity of the expression
    :param kernel_tokens: The basic tokens from which to build an expression
    :param weight_simple_expressions: Whether to correct for the issue that more complex
    expressions are more numerous.
    :return a sequence of kernel identifiers and optionally the multiplication token
    """
    # First decide length of expression
    n_summation_terms = np.random.randint(low=min_complexity, high=1 + max_complexity)
    # get the expression tokens, i.e. primitive kernels
    tokens = kernel_tokens or list(get_kernel_map().keys())

    if weight_simple_expressions:
        weights = compute_token_weights(tokens)
        probabilities = weights / np.sum(weights)
    else:
        probabilities = None

    return np.random.choice(tokens, n_summation_terms, replace=False, p=probabilities)


def make_kernel_from_label(
    expression: Sequence[str], num_dimensions: int, sigma: float, all_dims: bool = False,
    add_bias: bool = False) -> Kernel:
    """
    Build a gpflow kernel from an expression.

    :param expression: kernel expression
    :param num_dimensions: number of input dimensions
    :param sigma: breadth of priors in the log space
    :param all_dims: Whether to set kernel up with active_dims across all input dims or a random subset
    :param add_bias: Whether to add a bias to each subkernel
    :return the kernel corresponding to the given expression with default
            priors on the hyperparameters
    """

    sum_components = []
    kernel_map = get_kernel_map()

    for i, exp in enumerate(expression):
        if all_dims:
            active_dims = list(np.arange(num_dimensions))
        else:
            active_dims = choose_active_dims(num_dimensions)
        subkernel = construct_subkernel(exp, kernel_map=kernel_map, active_dims=active_dims, add_bias=add_bias)

        scaling = compute_prior_scaling(subkernel)
        set_default_priors_on_hyperparameters(subkernel, sigma=sigma, scaling=scaling)
        sum_components.append(subkernel)

    composite_kernel = Sum(sum_components) if len(sum_components) > 1 else sum_components[0]

    return composite_kernel


def compute_prior_scaling(kernel):
    """ Compute the scaling factor required to adjust the priors within a product kernel.
     The lengthscale should be reduced by a factor of scaling.
     The variance is unaffected as only one variance is active epr product kernel.
     """

    scaling = 1.0
    if isinstance(kernel, Product):
        n_bandwidth_kernels = sum(isinstance(k, BANDWIDTH_KERNELS) for k in kernel.kernels)
        scaling = np.maximum(1.0, n_bandwidth_kernels)

    return scaling


def construct_subkernel(
    subkernel, kernel_map, active_dims, fix_variance: bool = False, add_bias: bool = False) -> Kernel:
    """ Assemble gpflow kernel as a product of primitive kernels. Only the final entry has a trainable variance. """

    if MULTIPLICATION_TOKEN in subkernel:
        index = subkernel.index(MULTIPLICATION_TOKEN)
        left_exp = subkernel[:index]
        right_exp = subkernel[(index + 1):]
        left_kernel = construct_subkernel(left_exp, kernel_map, active_dims, fix_variance=True)
        right_kernel = construct_subkernel(
            right_exp, kernel_map, active_dims, fix_variance=False)
        gp_kernel = Product([left_kernel, right_kernel])
    else:  # Pure primitive kernel
        constructor = kernel_map[subkernel][0]
        gp_kernel = build_kernel(constructor, active_dims)

        if fix_variance and hasattr(gp_kernel, "variance"):
            if constructor not in [gf.kernels.Linear, ShiftedLinear]:
                # Linear kernels have multiple variance params so we should keep these trainable
                set_trainable(gp_kernel.variance, False)

        if add_bias:
            gp_kernel = gp_kernel + gf.kernels.Constant()

    return gp_kernel


def build_kernel(constructor, active_dims) -> gf.kernels.Kernel:
    """ Instantiate the gpflow kernel with appropriate active dims. """

    if constructor in [gf.kernels.Linear, ShiftedLinear]:
        gp_kernel = constructor(active_dims=active_dims, variance=np.ones(len(active_dims)))
    elif issubclass(constructor, gf.kernels.Stationary):
        gp_kernel = constructor(active_dims=active_dims, lengthscales=np.ones(len(active_dims)))
    elif constructor in [gf.kernels.Periodic, Periodic]:
        gp_kernel = constructor(active_dims=active_dims, lengthscales=np.ones(len(active_dims)),
                                period=np.ones(len(active_dims)))
    elif constructor is gf.kernels.White:
        gp_kernel = constructor(active_dims=active_dims)
    else:
        raise NotImplementedError('Unsupported kernel', constructor)

    return gp_kernel


def compute_token_weights(tokens) -> np.ndarray:
    """
    More complex product kernels are much more numerous than simple ones. To redress the balance we
    weight them such that the total weight is equal for a given number of products.
    """

    products = list(np.zeros(len(tokens)))
    weights = np.zeros(len(tokens))

    # First establish product counts for each token
    target = "\*"
    for i, t in enumerate(tokens):
        multis = re.findall(target, t)
        products[i] = len(multis)

    unique, counts = np.unique(products, return_counts=True)
    weights_dict = dict(zip(unique, 1 / counts))

    # Now translate products into weights
    for i, p in enumerate(products):
        weights[i] = weights_dict[p]

    return weights
