""" Tools for handling a list of gflow models """
from typing import List

import numpy as np
import gpflow as gf
import tensorflow as tf
from kitt.data.initialisation import perform_random_intialisation


def train_model(model: gf.models.GPModel, max_iter=1_000, method='bfgs'):
    """ Optimise a gpflow GPR model, defaults to BFGS. """
    opt = gf.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(),
        model.trainable_variables,
        method=method,
        options=dict(maxiter=max_iter, disp=1),
    )


def multi_train(model_list: List[gf.models.GPModel], max_iter: int):
    """ Initialise and train a list of models for max_iter number of steps"""

    for model in model_list:
        print("Initialising model", model.kernel)
        try:
            perform_random_intialisation(model, n_inits=1_000)
            train_model(model, max_iter)
        except tf.errors.InvalidArgumentError:  # If we hit a Cholesky error
            print("Ouch")
            try:
                perform_random_intialisation(model, n_inits=1_000)
                train_model(model, max_iter, 'CG')
            except tf.errors.InvalidArgumentError:  # If we hit a Cholesky error
                print("Aborting training - just find somewhere that works")
                if hasattr(model.likelihood, 'variance'):
                    model.likelihood.variance.assign(1.0)
                perform_random_intialisation(model, n_inits=100)


def multi_predict(model_list: List[gf.models.GPModel], x_test: np.ndarray):
    """ Get predictions from the mixture model """

    n_models = len(model_list)
    n_predict = x_test.shape[0]
    pred_means, pred_vars = np.zeros((n_predict, n_models)), np.zeros((n_predict, n_models))

    for (i, model) in enumerate(model_list):
        means, vars = model.predict_f(x_test)
        pred_means[:, i] = means[:, 0]
        pred_vars[:, i] = vars[:, 0]

    return pred_means, pred_vars


def evaluate_metrics(model_list: List[gf.models.GPModel], pred_means: np.ndarray, pred_vars: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, scaling_factor):
    """ Find the rmse and nlpd associated with a mixture of models.
        This method returns metrics per model """

    n_models = pred_means.shape[1]
    n_predict = y_test.shape[0]
    log_predictive_densities = np.zeros((n_predict, n_models))

    for (i, model) in enumerate(model_list):
        Fmean = pred_means[:, i][:, None]
        Fvar = pred_vars[:, i][:, None]
        lpd = model.likelihood.predict_log_density(x_test, Fmean, Fvar, y_test)
        log_predictive_densities[:, i] = tf.squeeze(lpd)
    
    pred_errors_per_model = pred_means - y_test
    rmse_per_model = tf.sqrt(tf.reduce_mean(pred_errors_per_model**2, axis=0)).numpy()  # vector of size n_models

    lpd_per_model = tf.math.reduce_mean(input_tensor=log_predictive_densities, axis=0, keepdims=True)
    lpd_per_model = tf.squeeze(lpd_per_model).numpy()

    rescaled_rmse = rmse_per_model * scaling_factor
    rescaled_lpd = lpd_per_model - np.log(scaling_factor)
    
    return rescaled_rmse.flatten(), rescaled_lpd.flatten()


def evaluate_model_averaged_metrics(model_list: List[gf.models.GPModel], pred_means: np.ndarray, pred_vars: np.ndarray,
                         y_test: np.ndarray, scaling_factor):
    """ Find the rmse and nlpd after model averaging over a mixture of GP models. """

    n_models = pred_means.shape[1]
    n_predict = y_test.shape[0]
    log_predictive_densities = np.zeros((n_predict, n_models))

    for (i, model) in enumerate(model_list):
        Fmean = pred_means[:, i][:, None]
        Fvar = pred_vars[:, i][:, None]
        lpd = model.likelihood.predict_log_density(Fmean, Fvar, y_test)
        log_predictive_densities[:, i] = tf.squeeze(lpd)

    # Get weights based upon LML values of the models
    weights = calculate_weights(model_list)

    model_average_mean = np.average(pred_means, weights=weights, axis=1)
    errors = model_average_mean - tf.squeeze(y_test)  # vector
    rmse = tf.sqrt(tf.reduce_mean(errors ** 2)).numpy()  # scalar

    lpd_contributions = log_predictive_densities + tf.math.log(weights)

    # vector
    ave_lpd = tf.math.reduce_logsumexp(input_tensor=lpd_contributions, axis=1, keepdims=True)
    
    # Reverse the effects of data normalisation
    rescaled_rmse = rmse * scaling_factor  # scalar
    rescaled_lpd = (tf.reduce_mean(ave_lpd) - np.log(scaling_factor)).numpy()  # scalar
    
    return rescaled_rmse.flatten(), rescaled_lpd.flatten()


def calculate_weights(model_list: List[gf.models.GPModel]) -> List:
    """
    Estimate weights for model selection.
    This is useful for combining the posterior distributions from multiple models. """

    lml_values = np.zeros(len(model_list))

    for (i, model) in enumerate(model_list):
        lml = model.maximum_log_likelihood_objective().numpy()
        lml_values[i] = lml if np.isfinite(lml) else -1e100

    return tf.nn.softmax(lml_values)  # todo could improve the model evidence estimate using eg BIC
