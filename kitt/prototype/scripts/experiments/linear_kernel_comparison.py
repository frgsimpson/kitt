"""
Script to run three variants of the linear kernel against UCI datasets.
Adapted from train_full_vocab_on_uci
"""

import json
from itertools import product
from typing import Dict, Union

import numpy as np
import argparse

from gpflow.models import GPR
from gpflow.utilities import read_values
from joblib import Parallel, delayed
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from kitt.config import LOG_DIR
from kitt.data.initialisation import perform_random_intialisation
from kitt.data.kitt_kernels import make_linear_noise_kernel
from kitt.data.multitrain import train_model, evaluate_metrics, multi_predict
from kitt.data.uci import dataset as uci_datasets
from kitt.data.uci.utils import subsample_dataset

import tensorflow as tf
tf.config.run_functions_eagerly(False)

N_MAX = 2000  # Largest dataset without subsampling
N_SPLITS = 10
NULL_RESULT = {
        "lpd": None,
        "rmse": None,
        "train_log_marginal_likelihood": None,
    }
KERNEL_NAMES = ['constantnoise', 'linearnoise', 'shiftedlinearnoise', 'offsetlinearnoise']
SPLIT_INDICES = [*range(N_SPLITS)]
# Run Naval and Kin8mn with args.n_jobs = 1
# DATASETS = ["Naval", "Kin8mn", "Boston", "Concrete", "Energy", "Power", "WineRed", "Yacht"]
DATASETS = ["Naval", "Boston", "Concrete", "Yacht"]


def safe_single_run(*args, **kwargs) -> Dict[str, Union[float, None]]:
    """
    Protect against any unforeseen cause of failure, such as loading the data
    (I'm looking at you, pandas)
    """

    try:
        results = single_run(*args, **kwargs)
    except Exception as e:
        print('Exception:', e)
        results = NULL_RESULT

    return results


def single_run(
        kernel_name: str,
        dataset_name: str,
        max_train_iters: int,
        test_train_split: float,
        split_index: int = 0
) -> Dict[str, Union[float, None]]:
    """
    Runs a single kernel on a single dataset reporting the Negative Log Predictive Density and the
    RMSE on the test set as well as the log marginal likelihood attained in training.
    If model training fails, null values are given for the metrics.

    :param kernel_name: The name of the kernel to evaluate. It will be instantiated by
        make_kernel_from_label
    :param dataset_name: The name of a UCI dataset to load (from kitt.data.uci.dataset)
    :param max_train_iters: The maximum number of training iterations
    :param test_train_split: The proportion of the data to keep for training the remainder being the
        test set.
    :param split_index: An integer indexing the test-train split (to facilitate reproducibility)

    :return: A dictionary of metrics measuring performance on the test set.
    """
    print(f"Running kernel {kernel_name} on {dataset_name}")
    dataset = getattr(uci_datasets, dataset_name)(split=split_index, prop=test_train_split)
    kernel = make_linear_noise_kernel(kernel_name, dataset.X_train.shape[1])

    if dataset.X_train.shape[0] >= N_MAX:
        dataset = subsample_dataset(dataset, N_MAX, args.test_train_split, seed=args.seed)
    model = GPR(data=(dataset.X_train, dataset.Y_train), kernel=kernel)
    perform_random_intialisation(model, n_inits=1_000)

    try:
        train_model(model, max_train_iters)
        print(f"Trained parameters", read_values(model))
    except InvalidArgumentError:
        return NULL_RESULT

    # Evaluate
    mean, var = multi_predict([model], dataset.X_test)
    rmse, lpd = evaluate_metrics(
        [model],
        mean,
        var,
        dataset.X_test,
        dataset.Y_test,
        scaling_factor=dataset.Y_std
    )
    lml_train = float(model.maximum_log_likelihood_objective().numpy())
    print(f"Marginal likelihood: ", lml_train)
    print(f"nlpd: ",  -1 * lpd)

    del model
    del dataset
    return {
        "lpd": float(lpd),
        "rmse": float(rmse),
        "train_log_marginal_likelihood": lml_train,
    }


def form_run_id(kernel_name: str, dataset_name: str) -> str:
    """ A simple function to make a unique id for the dataset-kernel combination """
    return f"{dataset_name}-{kernel_name}"


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    # Run experiments in parallel for a speed up.
    results_dict = dict(Parallel(n_jobs=args.n_jobs)(
        delayed(
            lambda k, d: (
                form_run_id(k, d),
                safe_single_run(k, d, args.max_train_iters, args.test_train_split, split_index=args.split_index)
            ))(kernel, dataset)
        for kernel, dataset in product(KERNEL_NAMES, DATASETS)
    ))
    print("RUNS COMPLETE")

    # Dump the results and the set up parameters to a json file.
    all_info = args.__dict__
    all_info.update(results_dict)

    save_dir = LOG_DIR / "linear_uci"
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = "uci_linear_results_split" + str(split_index)
    save_path = save_dir / filename

    with open(str(save_path) + ".json", "w") as json_file:
        json.dump(all_info, json_file, indent=2)


if __name__ == "__main__":
    for split_index in SPLIT_INDICES:
        parser = argparse.ArgumentParser()
        parser.add_argument("--split_index", type=int, default=split_index)
        parser.add_argument("--max_train_iters", type=int, default=1000)
        parser.add_argument("--test_train_split", type=float, default=0.9)
        parser.add_argument("--n_jobs", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        main(args)
