""" Tools for evaluating a captioning network. """
from typing import Optional, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.models import GPR, SGPR
from keras.preprocessing.text import Tokenizer

from kitt.data.kitt_kernels import make_linear_noise_kernel
from kitt.data.multitrain import multi_predict, evaluate_metrics
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import make_kernel_from_label
from kitt.prototype.evaluate_captions import process_kernel_expression, multi_evaluate
from kitt.prototype.evaluate_classifier import infer_top_n_expressions
from kitt.prototype.kitt_end_to_end import KITT
from kitt.networks.rnn_decoder import RNN_Decoder


def run_evaluation(
    generator: SampleGenerator, decoder, encoder, plot: bool = True, n_repeat: int = 10
) -> None:
    """ Perform multiple tests of the kernel identifier """

    nrows = 2
    ncols = n_repeat // nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 50))

    def last_nonzero(arr, axis, invalid_val=-1):
        """
        Finds the indices of the last occurrences of non-zero elements
        along an axis of a numpy array.
        """
        mask = arr != 0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    for i in range(nrows):
        for j in range(ncols):
            sample_batch, label_batch = generator.make_batch()
            image = sample_batch[0]
            sample = sample_batch[0]  # Select first example while retaining dimensions
            label = label_batch[0]

            # fetch the encoding of the expression labels
            # labels are 0-padded and have start-end tokens at beginning and end
            # real_label = label_batch[0][1:last_nonzero(label_batch, axis=1)[0]]
            expression_end = last_nonzero(label, axis=0)
            real_label = label[1:expression_end]  # Removes start, end, pad tokens

            real_expression = [generator.tokenizer.index_word[i] for i in real_label if i]
            real_kernel = process_kernel_expression(real_expression)

            predictions = multi_evaluate(sample, generator, decoder, encoder, process_caption=True)
            print("Real Kernel:", real_kernel)
            print("Predicted Kernel:", predictions)
            truth_string = r"$\mathbf{GT:" + real_kernel + "}$" + "\n"
            model_str = r"$\mathbf{Output:}$ "

            for n, (k_str, prob) in enumerate(predictions):  # List of tuples
                prob_str = "%.2f" % prob

                if k_str == real_kernel:
                    k_str = r"\mathbf{" + k_str + "}"

                if len(k_str) > 0:
                    k_str = "$" + k_str + "$"

                partial_string = k_str + " (" + prob_str + ")" + "\n"
                model_str = model_str + partial_string
                if n == 4:
                    break

            xlabel = truth_string + model_str
            if image.shape[0] > 1 and image.shape[1] > 1:
                axs[i, j].imshow(image)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            else:
                axs[i, j].plot(image.flatten())

            axs[i, j].set(xlabel=xlabel)
    fig.subplots_adjust(hspace=0.5)

    if plot:
        plt.show()


def make_models_from_expressions(expression_list, x_train, y_train, add_bias: bool = False,
                                 add_linear_noise: bool = False):
    """ Build a list of gpflow models based upon a list of expressions. """

    model_list = []
    n_dimensions = x_train.shape[1]
    n_units = x_train.shape[0]

    for expression in expression_list:
        kernel = make_kernel_from_label(
            expression,
            n_dimensions,
            sigma=1,
            all_dims=True,
            add_bias=add_bias
        )
        if add_linear_noise:
            linear_noise_kernel = make_linear_noise_kernel("shiftedlinearnoise", n_dimensions)
            kernel = kernel + linear_noise_kernel

        if n_units <= 2000:  # Large dataset revert to SGPR with n_inducing = 10% of original size
            model = GPR(data=(x_train, y_train), kernel=kernel)
        else:
            print('----Initialising SGPR Models with 10% inducing points')
            n_inducing = np.int(0.1*n_units)
            indices = np.random.choice(range(n_units), n_inducing)
            model = SGPR(data=(x_train, y_train), kernel=kernel, inducing_variable=x_train[indices])
        model_list.append(model)

    return model_list


def evaluate_prediction_vs_ground_truth(
        model: tf.keras.Model,
        tokenizer: Tokenizer,
        x_train: np.ndarray,
        y_train: np.ndarray,
        y_std: float,
        ground_truth_kernel: str,
        add_bias: bool = False,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Takes data generated by th ground truth kernel, uses the classifier to predict a kernel to model
    the raw data and builds GPR models with the ground truth and predicted kernels. These models are
    then evaluated on the provided train (and optionally test) data. Returns a dictionary of the
    results.

    :param model: The model used to make predictions of the appropriate kernel from data.
    :param tokenizer: The tokenizer used to map kernel names to usable tokens/indices and vice versa
    :param x_train: Training data (x values) for GPR.
    :param y_train: Training data (y values) for GPR.
    :param y_std: The standard deviation of the y data (as used in normalisation).
    :param ground_truth_kernel: The descriptor of the ground truth kernel (the kernel used to
        generate the data)
    :param add_bias: whether to add a bias to each (sub)kernel when instantiating the kernel.
    :param x_test: Optional test dataset (x values). If None test metrics will be None.
    :param y_test: Optional test dataset (x values). If None test metrics will be None.
    """
    predicted_kernels, _ = infer_top_n_expressions(
        model,
        tokenizer,
        x_train,
        y_train,
        n_expressions=1,
        kernel_only_vocab=not isinstance(model, (RNN_Decoder, KITT))
    )
    predicted_kernel = predicted_kernels[0]
    train_rmse, train_lpd, train_lml, test_rmse, test_lpd = compute_metrics_for_kernels(
        [[ground_truth_kernel], predicted_kernel], x_train, y_train, y_std, add_bias, x_test, y_test
    )

    return {
        "ground_truth": {
            "kernel": ground_truth_kernel,
            "lml": train_lml[0],
            "train_rmse": train_rmse[0],
            "train_lpd": train_lpd[0],
            "test_rmse": test_rmse[0],
            "test_lpd": test_lpd[0]
        },
        "prediction": {
            "kernel": predicted_kernel[0],
            "lml": train_lml[1],
            "train_rmse": train_rmse[1],
            "train_lpd": train_lpd[1],
            "test_rmse": test_rmse[1],
            "test_lpd": test_lpd[1]
        }
    }


def compute_metrics_for_kernels(
        kernel_descriptors: Sequence[Sequence[str]],
        x_train: np.ndarray,
        y_train: np.ndarray,
        y_std: float,
        add_bias: bool = False,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
) -> Tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[Optional[float]],
    Sequence[Optional[float]]
]:
    """
    Takes kernel descriptors, builds GPR models with them and returns metrics on the provided train
    (and optionally test) data.

    :param kernel_descriptors: A list of kernel labels identifying the kernels to build.
    :param x_train: Training data (x values) for GPR.
    :param y_train: Training data (y values) for GPR.
    :param y_std: The standard deviation of the y data (as used in normalisation).
    :param add_bias: whether to add a bias to each (sub)kernel when instantiating the kernel.
    :param x_test: Optional test dataset (x values). If None test metrics will be None.
    :param y_test: Optional test dataset (x values). If None test metrics will be None.
    """
    models = make_models_from_expressions(
        kernel_descriptors, x_train, y_train, add_bias
    )
    train_pred_mean, train_pred_var = multi_predict(models, x_train)
    train_rmse, train_lpd = evaluate_metrics(
        models, train_pred_mean, train_pred_var, x_train, y_train, y_std
    )
    lml = [float(m.maximum_log_likelihood_objective().numpy()) for m in models]

    if x_test:
        assert y_test is not None
        test_pred_mean, test_pred_var = multi_predict(models, x_test)
        test_rmse, test_lpd = evaluate_metrics(models, test_pred_mean, test_pred_var, x_test, y_test, y_std)
    else:
        test_rmse, test_lpd = [None, None], [None, None]

    return train_rmse, train_lpd, lml, test_rmse, test_lpd
