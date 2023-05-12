"""
Takes captions produced by KITT (typically from running run_kitt_on_uci.py) builds the GPR models,
trains and evaluates the models on the UCI datasets. Training is parallelised for speed but note
that there is a trade off where greater parallelism will require more memory. You can then decide
to run on CPU where parallelism may be greater or on GPU where memory is more likely to be a
limiting factor.

Results are saved in JSON files which can then be aggregated using aggregate_results.py.
"""

import argparse
import json
from collections import defaultdict
from itertools import product
from datetime import datetime
from typing import Dict, List, Type
from joblib import delayed, Parallel

import numpy as np
import tensorflow as tf

from gpflow.utilities import parameter_dict

from kitt.config import LOG_DIR
from kitt.data.multitrain import (
    multi_train,
    multi_predict,
    evaluate_metrics,
    evaluate_model_averaged_metrics
)
from kitt.data.uci.utils import subsample_dataset, ExperimentName
from kitt.prototype.evaluate import make_models_from_expressions
from kitt.utils.misc import get_args_string
from kitt.utils.training import Dataset
from kitt.data.uci import dataset as uci_datasets

# Run Naval and Kin8mn with args.n_jobs = 1

DATASETS = ["Boston", "Concrete", "Energy", "Kin8mn", "Power", "WineRed", "Yacht", "Naval"]
#"Kin8mn" "Naval"
LINEAR_NOISE = True  # whether to add the linear noise kernel as default
N_SPLITS = 10
SPLIT_INDICES = [*range(N_SPLITS)]


def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)

def make_serialisable_parameter_dict(model: tf.Module) -> Dict[str, List[float]]:
    return {k: v.numpy().tolist() for k, v in parameter_dict(model).items()}

def experiment_name(
    date_str,
    dataset_name,
    encoder_name,
    decoder_name,
    split_index,
    n_models,
    train_test_split,
    subset,
    max_iter,
    using_captions,
    **kwargs
):
    return (
        ExperimentName(date_str)
        .add("dataset", dataset_name)
        .add("encoder", encoder_name)
        .add("decoder", decoder_name)
        .add("split", split_index)
        .add("n_models", n_models)
        .add("frac", train_test_split)
        .add("subset", subset)
        .add("max_iter", max_iter)
        .add("captioning", using_captions)
        .get()
    )

def baseline_experiment_name(
    date_str,
    dataset_name,
    split_index,
    train_test_split,
    subset,
    max_iter,
    **kwargs
):
    return (
        ExperimentName(date_str)
        .add("dataset", dataset_name)
        .add("split", split_index)
        .add("frac", train_test_split)
        .add("subset", subset)
        .add("max_iter", max_iter)
        .get()
    )

def single_run(
    dataset_name: str,
    split_index: int,
    train_test_split: float,
    expression_list: List[str],
    expression_weights: List[float],
    inference_time: float,
    max_iter: int,
    date_str: str,
    encoder_name: str,
    decoder_name: str,
    expressions_path: str,
    subset: bool = True,
    baseline: bool = False
):
    print(f"Running on {dataset_name} (Split {split_index})")

    dataset = get_dataset_class(dataset_name)(split=split_index, prop=train_test_split)
    if subset:
        dataset = subsample_dataset(dataset, n_max=2000, train_prop=train_test_split)
    model_list = make_models_from_expressions(expression_list, dataset.X_train, dataset.Y_train, add_linear_noise=LINEAR_NOISE)
    multi_train(model_list, max_iter)
    multi_means, multi_vars = multi_predict(model_list, dataset.X_test)
    rmse_per_model, lpd_per_model = evaluate_metrics(model_list, multi_means, multi_vars, dataset.X_test,
                                                     dataset.Y_test, dataset.Y_std)
    rmse_joint, lpd_joint = evaluate_model_averaged_metrics(
        model_list, multi_means, multi_vars, dataset.Y_test, dataset.Y_std
    )
    uniform_avg_test_rmse = np.mean(rmse_per_model)
    uniform_avg_test_nlpd = -1 * np.mean(lpd_per_model)
    norm_expression_weights = tf.keras.utils.normalize(expression_weights, order=1).flatten()
    wt_avg_test_rmse = np.average(rmse_per_model, weights=norm_expression_weights)
    wt_avg_test_nlpd = -1 * np.average(lpd_per_model, weights=norm_expression_weights)
    model_ave_rmse = np.round(rmse_joint[0], 6)
    model_ave_nlpd = np.round(-1 * lpd_joint[0], 6)

    per_model_metrics = {'per_model_rmse': np.round(rmse_per_model, 6).tolist(),
                         'per_model_nlpd': np.round(-lpd_per_model, 6).tolist()}
    uni_avg_test_metrics = {'avg_test_rmse': np.round(uniform_avg_test_rmse, 6),
                            'avg_test_nlpd': np.round(uniform_avg_test_nlpd, 6)}
    wt_avg_test_metrics = {'wt_test_rmse': np.round(wt_avg_test_rmse, 6),
                           'wt_test_nlpd': np.round(wt_avg_test_nlpd, 6)}
    model_avg_test_metrics = {'model_avg_test_rmse': model_ave_rmse,
                              'model_avg_test_nlpd': model_ave_nlpd}
    weights_dict = {'expression_weights': list(expression_weights)}
    normed_weights_dict = {'normed_weights': list(norm_expression_weights)}
    wall_clock_time = {'inference_time': inference_time}

    model_parameters = {
        f"model-{i}-parameters": make_serialisable_parameter_dict(m)
        for i, m in enumerate(model_list)
    }
    exp_info = {
        "kernel": expression_list,
        "date_str": date_str,
        "encoder_name": encoder_name,
        "decoder_name": decoder_name,
        "expressions_path": expressions_path,
        "split_index": split_index,
        "dataset_name": dataset_name,
        "n_models": len(expression_list),
        "subset": subset,
        "max_iter": max_iter,
        "using_captions": True,
        "train_test_split": train_test_split
        }

    # One dict with all info + averaged metrics
    experiment_dict = {**exp_info,
                       **per_model_metrics,
                       **uni_avg_test_metrics,
                       **wt_avg_test_metrics,
                       **model_avg_test_metrics,
                       **weights_dict,
                       **normed_weights_dict,
                       **wall_clock_time,
                       **model_parameters}

    print(dataset_name, ': Model Avg. Test RMSE:', model_ave_rmse)
    print(dataset_name, ': Model Avg. Test NLPD:', model_ave_nlpd)
    
    if baseline is False:
        results_path = LOG_DIR / date_str / "results" / experiment_name(**exp_info)
        filename = f"{results_path}_multi_train_results.json"
    else:
        results_path = LOG_DIR / date_str / "results" / baseline_experiment_name(**exp_info)
        filename = f"{results_path}_rbf_baseline_results.json"
     
    with open(filename, "w") as fp:
        json.dump(experiment_dict, fp, indent=4)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    date_str = args.date_str or datetime.now().strftime('%b%d')
    save_dir = LOG_DIR / date_str / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.rbf_baseline_experiment:
        expressions = defaultdict(lambda: ([["RBF"]], [1.0], 0.0))
        expressions_save_path = None
    else:
        if args.decoder_name is None:
            # Use classifier-transformer only
            suffix = "expressions" + "_max_prod_" + str(args.max_terms) + ".json"
            expressions_save_path = save_dir / suffix
        else:
            # Use full-KITT
            suffix = "captions" + "_max_prod_" + str(args.max_terms) + ".json"
            expressions_save_path = save_dir / suffix
        with open(expressions_save_path, "r") as f:
            expressions = json.load(f)


    args_save_path = save_dir / "run_multi_train_parallel_arguments.txt"
    with open(str(args_save_path), "w") as file:
        file.write(get_args_string(args))

    print(
        "Training GPR in parallel. GPUs may run out of memory, use CPU if this happens. "
        "To use CPU only, set environment variable CUDA_VISIBLE_DEVICES=-1"
    )

    Parallel(args.n_jobs)(
        delayed(single_run)(
            dsn,
            i,
            args.test_train_split,
            *expressions[f"{dsn.lower()}-{i}"],
            args.max_iters,
            date_str,
            args.encoder_name,
            args.decoder_name,
            str(expressions_save_path),
            args.subset,
            args.rbf_baseline_experiment
        ) for dsn, i in product(DATASETS, SPLIT_INDICES)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.add_argument("--max_iters", type=int, default=10_000)
    parser.add_argument("--test_train_split", type=float, default=0.9)
    parser.add_argument("--rbf_baseline_experiment", action="store_true", default=False, help="Use RBF for all datasets.")
    parser.add_argument("--encoder_name", type=str, default="classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs")
    parser.add_argument("--decoder_name", type=str, default=None)  # "decoder-transformer")
    parser.add_argument("--max_terms", type=int, default=2, help="For the moment must match max_terms used in training.")
    parser.add_argument("--date_str", type=str, default="May17")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.set_defaults(subset=True)
    arguments = parser.parse_args()
    main(arguments)
