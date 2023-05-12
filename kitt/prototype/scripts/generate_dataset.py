"""
Generate a dataset for training and evaluating an attention network.

The output is a TGZ archive with the following content:
- train.h5: contains training set samples and corresponding labels from random kernel expressions
  with different hyperparameter initialisations;
- test.h5: contains test set samples and corresponding labels from random kernel expressions with
  different hyperparameter initialisations;
- tokenizer.pickle: serialised tokenizer object

The archive is written in the datasets directory.
"""
import argparse
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import tensorflow as tf
from tqdm import tqdm

from kitt.config import DATASET_DIR
from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import load_default_coords, load_random_coords
from kitt.data.tokeniser import KernelTokenizer
from kitt.prototype.configs.args import parse_command_line_args
from kitt.prototype.configs.configs import load_config
from kitt.utils.misc import yes_or_no

N_RANDOM_SAMPLES = 64
N_RANDOM_SAMPLES_FOR_CAPTIONS = 256


def get_arguments() -> argparse.Namespace:
    """ Parse and validate script arguments """

    user_arguments = parse_command_line_args()
    config = load_config(user_arguments.base_config)
    user_overrides = vars(user_arguments)
    for arg, value in config.items():
        config[arg] = value if user_overrides[arg] is None else user_overrides[arg]
    
    default_output_dir = f"{N_RANDOM_SAMPLES}_{config['num_dimensions']}d_max_exp_{config['max_expression']}_prod_{config['max_products']}"
    print('Saving to ', default_output_dir)
    if config["make_captions"]:
        default_output_dir = default_output_dir + '_captions'
        
    if config["include_x_with_samples"]:
        default_output_dir = default_output_dir + '_random_inputs'
    else:
        default_output_dir = default_output_dir + '_grid_inputs'
        
    config["output_dir"] = default_output_dir

    parsed_args = argparse.Namespace(**config)
 
    if parsed_args.max_expression > 1:
        assert (
            parsed_args.make_captions
        ), "Must use caption labels for expressions longer than 1 kernel."
    assert parsed_args.min_expression <= parsed_args.max_expression
    assert parsed_args.num_hyper_randomisations <= parsed_args.num_samples_per_expression
    assert not parsed_args.num_samples_per_expression % parsed_args.num_hyper_randomisations
    assert 0 < parsed_args.train_ratio < 1.0

    return parsed_args


@dataclass
class Resolution:
    x: int
    y: int


def write_features_and_labels_to_file(
    file_path: Path,
    tokenizer: KernelTokenizer,
    num_expressions: int,
    min_expression: int,
    max_expression: int,
    make_captions: bool,
    num_dimensions: int,
    sigma: float,
    resolution: Resolution,
    num_samples_per_expression: int,
    num_hyper_randomisations: int,
    use_fft: bool,
    include_x_with_samples: bool,
) -> None:
    with h5py.File(file_path, 'a') as hf:

        # store some metadata, useful when evaluating model
        metadata = {
            "num_expressions": num_expressions,
            "num_samples_per_expression": num_samples_per_expression,
            "num_hyper_randomisations": num_hyper_randomisations,
            "min_expression": min_expression,
            "max_expression": max_expression,
            "num_dimensions": num_dimensions,
            "sigma": sigma,
            "x_resolution": resolution.x,
            "y_resolution": resolution.y,
            "fft": use_fft,
            "include_x_with_samples": include_x_with_samples,
        }
        for k, v in metadata.items():
            hf.attrs[k] = v

        if include_x_with_samples:
            coords = load_random_coords(resolution.x, n_dims=num_dimensions)
        else:
            coords = load_default_coords(x_resolution=resolution.x, y_resolution=resolution.y)

        print(f"Generating samples from {num_expressions} kernel expressions")

        sample_generator = SampleGenerator(
            x_values=coords,
            x_resolution=resolution.x,
            batch_size=num_samples_per_expression,
            min_expression=min_expression,
            max_expression=max_expression,
            sigma=sigma,
            use_fft=use_fft,
            standardise=True,
            tokenizer=tokenizer,
            all_dims=True,
            make_captions=make_captions,
            include_x_with_samples=include_x_with_samples,
            iterations_per_epoch=num_expressions * num_hyper_randomisations,
        )

        for _ in tqdm(range(num_expressions * num_hyper_randomisations)):
            samples, labels = sample_generator.make_batch()
            if ("X" in hf) and ("Y" in hf):
                # append samples and labels to their respective datasets
                hf["X"].resize((hf["X"].shape[0] + samples.shape[0]), axis=0)
                hf["Y"].resize((hf["Y"].shape[0] + labels.shape[0]), axis=0)
                hf["X"][-samples.shape[0] :] = samples
                hf["Y"][-labels.shape[0] :] = labels
            else:
                # First expression: create datasets for samples and labels
                hf.create_dataset(
                    "X",
                    data=samples,
                    # compression="gzip",
                    # compression_opts=9,
                    chunks=True,
                    maxshape=(None,) + samples.shape[1:],
                )
                hf.create_dataset(
                    "Y",
                    data=labels,
                    # compression="gzip",
                    # compression_opts=9,
                    chunks=True,
                    maxshape=(None,) + labels.shape[1:],
                )
        print("...Done.")


def main(args: argparse.Namespace) -> None:

    tf.random.set_seed(args.random_seed)

    output_dir = DATASET_DIR / args.output_dir
    if output_dir.exists():
        overwrite = yes_or_no(f"{output_dir} will be overwritten. Proceed?")
        if overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            quit()

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = output_dir.with_suffix(".tar.gz")
    try:
        archive_name.unlink()
    except FileNotFoundError:
        pass

    file_paths = [output_dir / "train.h5", output_dir / "test.h5"]
    # todo fix minor issue: datasets themselves are split into train/test within each train/test file
    num_expressions = [
        int(args.num_expressions * args.train_ratio),
        int(args.num_expressions * (1 - args.train_ratio)) + 1,
    ]
    sigmas = [args.train_sigma, args.test_sigma]

    if args.include_x_with_samples: # random inputs
        n_random = N_RANDOM_SAMPLES_FOR_CAPTIONS if args.make_captions else N_RANDOM_SAMPLES
        res = Resolution(x=n_random, y=1)
    else:
        res = (
            Resolution(x=16, y=16) if args.num_dimensions == 2 else Resolution(x=256, y=1)
        )  # x res used in multidim case

    # Product kernels can be explicitly embedded in vocabulary, may prove easier to learn
    vocab = get_unique_product_kernels(max_terms=args.max_products)
    kernel_expression_tokenizer = KernelTokenizer(vocabulary=vocab)

    for (file_path, num_exprs, sigma) in zip(file_paths, num_expressions, sigmas):
        write_features_and_labels_to_file(
            file_path=file_path,
            tokenizer=kernel_expression_tokenizer,
            num_expressions=num_exprs,
            min_expression=args.min_expression,
            max_expression=args.max_expression,
            make_captions=args.make_captions,
            num_dimensions=args.num_dimensions,
            sigma=sigma,
            resolution=res,
            num_samples_per_expression=args.num_samples_per_expression,
            num_hyper_randomisations=args.num_hyper_randomisations,
            use_fft=args.use_fft,
            include_x_with_samples=args.include_x_with_samples,
        )

    # serialise tokeniser
    with open(output_dir / "tokenizer.pickle", "wb") as f:
        pickle.dump(kernel_expression_tokenizer, f)

    shutil.make_archive(f"{output_dir}", "gztar", DATASET_DIR, output_dir.name)
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
