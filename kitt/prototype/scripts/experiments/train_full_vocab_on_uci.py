"""
Script to run each kernel in a considered vocabulary against UCI datasets.

This is in part a benchmarking exercise but also enables us to see if KITT or the classifier is
suggesting kernels which we know to work without having to rerun the training each time.
"""

import json
from itertools import product
from typing import Dict, Union

import numpy as np
import argparse
from pathlib import Path

from gpflow.models import GPR
from gpflow.utilities import read_values
from joblib import Parallel, delayed
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from kitt.data.kernels import get_unique_product_kernels
from kitt.data.initialisation import perform_random_intialisation
from kitt.data.multitrain import train_model, evaluate_metrics, multi_predict
from kitt.data.sampler.utils import make_kernel_from_label
from kitt.data.uci import dataset as uci_datasets
from kitt.data.uci.utils import subsample_dataset, COMPARISON_DATASETS

import tensorflow as tf

tf.config.run_functions_eagerly(True)

N_MAX = 2000
NULL_RESULT = {
        "lpd": None,
        "rmse": None,
        "train_log_marginal_likelihood": None,
    }


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
    kernel = make_kernel_from_label(
        [kernel_name],
        dataset.X_train.shape[1],
        sigma=1.0,
        all_dims=True
    )
    if dataset.X_train.shape[0] >= N_MAX:
        dataset = subsample_dataset(dataset, N_MAX, args.test_train_split, seed=args.seed)

    model = GPR(data=(dataset.X_train, dataset.Y_train), kernel=kernel)
    perform_random_intialisation(model)

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
    kernel_names = get_unique_product_kernels(max_terms=args.max_products)

    # Run experiments in parallel for a speed up.
    results_dict = dict(Parallel(n_jobs=args.n_jobs)(
        delayed(
            lambda k, d: (
                form_run_id(k, d),
                safe_single_run(k, d, args.max_train_iters, args.test_train_split)
            ))(kernel, dataset)
        for kernel, dataset in product(kernel_names, COMPARISON_DATASETS)
    ))
    print("RUNS COMPLETE")

    # Dump the results and the set up parameters to a json file.
    all_info = args.__dict__
    all_info.update(results_dict)

    path = Path(__file__).parent if args.log_dir is None else Path(args.log_dir)

    with open(str(path / args.filename) + ".json", "w") as json_file:
        json.dump(all_info, json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_products", type=int, default=1)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--filename", type=str, default="uci_GPR_benchmark_results_hetcomb_ard_rbftrainx2")
    parser.add_argument("--max_train_iters", type=int, default=1000)
    parser.add_argument("--test_train_split", type=float, default=0.9)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=223)
    args = parser.parse_args()

    main(args)
