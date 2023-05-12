"""
Loads a trained KITT or Classifier model and runs it on UCI datasets to generate captions which are
then saved to disk.
To train and evaluate the proposed kernels for GPR on the UCI datasets, use
run_multi_train_parallel.py

This script will run fastest when GPU is available.
"""

import argparse
import json
from time import time
from datetime import datetime
from typing import Optional, Type, Union

import numpy as np
import tensorflow as tf

from kitt.config import LOG_DIR
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.tokeniser import KernelTokenizer
from kitt.data.uci.utils import subsample_dataset
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.networks.transformer.transformer_decoder import TransformerDecoder
from kitt.prototype.evaluate_captions import infer_top_n_captions
from kitt.prototype.evaluate_classifier import infer_top_n_expressions
from kitt.prototype.get_models import (
    get_trained_captioning_network,
    get_trained_classification_network
)
from kitt.prototype.scripts.experiments.run_multi_train_parallel import N_SPLITS
from kitt.utils.misc import get_args_string
from kitt.utils.training import Dataset
from kitt.data.uci import dataset as uci_datasets

# WARNING: It might be necessary to pre-download Kin8mn if you don't have it already as loading it on the fly might 
# cause the script to crash.

DATASETS = ["Boston", "Concrete", "Energy", "Kin8mn", "Power", "WineRed", "Yacht", "Naval"]
SPLIT_INDICES = [*range(N_SPLITS)]


def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)


def get_kernel_expressions(
        encoder: tf.keras.Model,
        decoder: Optional[Union[RNN_Decoder, TransformerDecoder]],
        sample_generator: Optional[SampleGenerator],
        dataset: Dataset,
        tokenizer: Optional[KernelTokenizer],
        num_gp_models: int
):
    start = time()
    if decoder is not None:
        assert sample_generator is not None, "Inference with KITT requires a sample generator."
        expression_list, weights = infer_top_n_captions(
            encoder, decoder, sample_generator, dataset.X_train, dataset.Y_train, num_gp_models
        )
    else:
        assert tokenizer is not None, "Inference with classifier requires a tokenizer."
        expression_list, weights = infer_top_n_expressions(
            encoder, tokenizer, dataset.X_train, dataset.Y_train, num_gp_models
        )
    inference_time = np.round(time() - start, 4)
    return expression_list, weights, inference_time


def main(args: argparse.Namespace):
    date_str = args.date_str or datetime.now().strftime('%b%d')
    save_dir = LOG_DIR / date_str / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    args_save_path = save_dir / "run_kitt_on_uci_arguments.txt"
    with open(str(args_save_path), "w") as file:
        file.write(get_args_string(args))
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.full_kitt:
        encoder, decoder, sample_generator = get_trained_captioning_network(
            args.encoder_name,
            args.decoder_name,
            args.max_terms
        )
        tokenizer = None
        suffix = "captions" + "_max_prod_" + str(args.max_terms) + ".json"
        expressions_save_path = save_dir / suffix
    else:
        encoder, tokenizer = get_trained_classification_network(args.encoder_name, args.max_terms)
        decoder = None
        sample_generator = None
        suffix = "expressions" + "_max_prod_" + str(args.max_terms) + ".json"
        expressions_save_path = save_dir /  suffix
    print("MODEL LOAD COMPLETED")
    expressions = dict()
    for dataset in DATASETS:
        ds = get_dataset_class(dataset)(prop=args.test_train_split)
        for split_index in SPLIT_INDICES:
            ds.apply_split(split_index)

            if args.subset:
                ds = subsample_dataset(ds, n_max=2000, train_prop=args.test_train_split)

            key = f"{ds.name.lower()}-{split_index}"
            expressions[key] = get_kernel_expressions(
                encoder, decoder, sample_generator, ds, tokenizer, args.num_models_per_dataset
            )
            print(f"Inference complete for {dataset} (split {split_index})")
            
    with open(str(expressions_save_path), "w") as f:
        json.dump(expressions, f, indent=4)
    print("Caption Inference Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--test_train_split", type=float, default=0.9)
    parser.add_argument("--encoder_name", type=str, default="classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs")
    parser.add_argument("--decoder_name", type=str, default="decoder-rnn_147686_params")
    parser.add_argument("--max_terms", type=int, default=2, help="For the moment must match max_terms used in training.")
    parser.add_argument("--num_models_per_dataset", type=int, default=3)
    parser.add_argument("--date_str", type=str, default=None)
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.add_argument("--full_kitt", dest="full_kitt", action="store_true")
    parser.add_argument("--classifier_only", dest="full_kitt", action="store_false")

    parser.set_defaults(subset=True, full_kitt=True)

    arguments = parser.parse_args()
    main(arguments)
