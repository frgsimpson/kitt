""" Defines priors for various parameters. """

import gpflow as gf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

DEFAULT_SIGMA = 1.0  # Breadth of prior in log space


def _get_tf_const(value: float) -> tf.Tensor:
    return tf.constant(value, dtype=gf.default_float())


def set_default_priors_on_hyperparameters(
        module: gf.base.Module, replace_all: bool = False, sigma: float = DEFAULT_SIGMA,
        scaling: float = 1.0) -> int:
    """Assigns default priors to any trainable hyperparameters which lack priors.

    :param module: Any GPflow module, such as a Kernel or GPModel
    :param replace_all: Whether to override existing priors assigned to the parameter
    :param sigma: Breath of the prior in log space
    :param scaling: adjustment required for components of product kernels
    :return number of parameters which have been assigned default priors
    """

    n_defaults = 0
    param_dict = gf.utilities.leaf_components(module)

    for path, parameter in param_dict.items():
        
        if path == 'SGPR.inducing_variable.Z':
            set_prior = False
            print('Will not optimise inducing locations at initialisation')
        else:
            set_prior = parameter.trainable and (replace_all or parameter.prior is None)

        if set_prior:
            parameter.prior = load_default_prior(parameter, path, sigma, scaling)
            n_defaults += 1

    return n_defaults


def load_default_prior(
    parameter: gf.Parameter, path: str = "default", sigma: float = DEFAULT_SIGMA,
        scaling: float = 1.0) -> tfp.distributions.Distribution:
    """Assigns default priors to a parameter based upon their path and choice of transform.
    If parameter is unknown, a unit normal distribution is provided.

    :param parameter: The parameter whose prior distribution we wish to determine
    :param path: The path is helpful in determining the nature of the parameter
    :param sigma: Breadth of the prior in log space
    :param scaling: adjustment required for components of product kernels
    :return: Prior probability distribution
    """

    transform_name = "null" if parameter.transform is None else parameter.transform.name

    if path.endswith("variance"):
        prior = tfp.distributions.LogNormal(loc=np.float64(-2.0), scale=np.float64(sigma))
    elif path.endswith("period"):
        # todo custom period prior based upon extent of training data. use 0.2 for now.
        typical_period = 0.2 * scaling
        log_period = np.float64(np.log(typical_period))
        scale = np.float64(sigma * scaling ** 0.25)  # See scaling_priors script for heuristics
        prior = tfp.distributions.LogNormal(loc=log_period, scale=scale)
    elif path.endswith("shifts"):  # Prior set up to produce a suitable linear noise distribution
        prior = tfp.distributions.Cauchy(loc=np.float64(0.0), scale=np.float64(5.0))
    elif path.endswith("variances"):
        m = np.float64(1 / len(parameter.numpy()))
        prior = tfp.distributions.LogNormal(loc=np.log(m), scale=np.float64(sigma))
    elif path.endswith("lengthscales") or transform_name in [
        "exp",
        "softplus",
    ]:  # todo custom lengthscale prior based upon extent of training data. use 1.0 for now.
        typical_length = 1.0 * scaling
        log_lengthscale = np.float64(np.log(typical_length))
        scale = np.float64(sigma * scaling ** 0.25)  # See scaling_priors script for heuristics
        prior = tfp.distributions.LogNormal(loc=log_lengthscale, scale=scale)
    else:
        # Default to unit Gaussian prior
        prior = tfp.distributions.Normal(loc=np.float64(0.0), scale=np.float64(1.0))

    return prior
