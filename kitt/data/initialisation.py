import gpflow
from gpflow.utilities import multiple_assign, read_values, set_trainable
import tensorflow as tf

from kitt.data.priors import set_default_priors_on_hyperparameters
from kitt.data.sampler.utils import randomise_hyperparameters


def perform_random_intialisation(model: gpflow.models.GPModel, n_inits: int = 1_000) -> None:
    """ Update a GP model with the best parameters from a random initialisation. """

    if model.likelihood.variance is not None:
        initialise_likelihood_variance(model)
    set_default_priors_on_hyperparameters(model)

    best_param_values, max_likelihood = find_best_random_sample(
        model, n_inits
    )
    multiple_assign(model, best_param_values)
    if model.likelihood.variance is not None:
        set_trainable(model.likelihood.variance, True)

    print("Best log likelihood:", max_likelihood)
    print("Initialised parameters:", best_param_values)


def initialise_likelihood_variance(model):
    """ While finding the best set of hyperparams, its usually best to fix the likelihood variance. """

    model.likelihood.variance.assign(0.25)
    set_trainable(model.likelihood.variance, True)


def find_best_random_sample(model, n_inits: int = 2000):
    """
    Find the best set of hyperparameters for a model by drawing samples from their priors.
    """

    @tf.function
    def evaluate_likelihood_of_random_parameters() -> tf.Tensor:
        """
        Provide a speedup via ``tf.function magic``.  The `loss_closure` function operates on the
        model itself.
        """
        randomise_hyperparameters(model)
        return model.maximum_log_likelihood_objective()

    max_log_likelihood = -1e100
    n_fails = 0
    best_parameters = read_values(model)

    for i in range(n_inits):
        try:
            likelihood = evaluate_likelihood_of_random_parameters().numpy()
        except tf.errors.InvalidArgumentError:
            n_fails += 1
            likelihood = -1e100
            if n_fails < 5:
                print("Pathological parameters at initialization ", i, ":", read_values(model))
        if likelihood > max_log_likelihood:
            max_log_likelihood = likelihood
            best_parameters = read_values(model)
            print("New max likelihood:", max_log_likelihood, "at init ", i, "of", n_inits)

    return best_parameters, max_log_likelihood
