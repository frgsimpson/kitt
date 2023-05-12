import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import tensorflow as tf

from kitt.config import BASE_PATH, MODEL_SAVE_DIR
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import make_kernel_from_label, load_random_coords
from kitt.data.tokeniser import START_TOKEN, END_TOKEN, PAD_TOKEN
from kitt.prototype.evaluate import compute_metrics_for_kernels
from kitt.utils.save_load_models import load_kitt


def get_arguments() -> Namespace:
    """ Parse and validate standard experimental arguments """

    parser = ArgumentParser()

    parser.add_argument(
        "--num_dimensions",
        type=int,
        help="Dimensionality of the data",
        default=2,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Number of x values in each dimension",
        default=64,
    )
    parser.add_argument(
        "--x_scaling",
        type=float,
        help="Scaling of the random values samples as x values (initially in the unit interval).",
        default=5,
    )
    parser.add_argument(
        "--max_products",
        type=int,
        help="The maximal number of terms permissable in a product kernel.",
        default=3,
    )
    parser.add_argument(
        "--model_load_path",
        default=MODEL_SAVE_DIR,
        type=Path,
        help="The directory from which to load the model. "
             "Note that there is an expected directory strueture expected when loading KITT.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Directory where to save model and any relevant logs",
        default=BASE_PATH / "results",
    )
    parser.add_argument(
        "--kitt_encoder",
        help="Encoder identifier when using full KITT. Used to load the right model.",
        default="classifier-transformer",
        type=str
    )
    parser.add_argument(
        "--kitt_decoder",
        help="Decoder identifier when using full KITT. Used to load the right model.",
        default="decoder-transformer",
        type=str
    )
    parser.add_argument(
        "--max_expression_length",
        help="The maximal number of kernels in the sum expression to allow when using KITT.",
        default=3,
        type=int
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for Numpy",
        default=1234
    )

    return parser.parse_args()


def single_kernel_evaluation(
        kernel_description: str,
        sample_generator: SampleGenerator,
        model: tf.keras.Model
):
    kernel = make_kernel_from_label(
        [kernel_description],
        sample_generator.ndims,
        sample_generator.sigma,
        all_dims=sample_generator.all_dims,
        add_bias=sample_generator.add_bias
    )
    example_data = sample_generator.make_batch_of_samples(kernel)
    sample_data = example_data[0]
    sample_x = sample_data[..., :-1]
    sample_y = sample_data[..., -1:]
    y_std = sample_y.std()
    caption, _ = model.get_kernel_expression_for_sample(
        sample_data, temperature=-100,
    )
    caption = [elem for elem in caption[0].split() if elem not in [START_TOKEN, PAD_TOKEN, END_TOKEN]]
    train_rmse, train_lpd, train_lml, test_rmse, test_lpd = compute_metrics_for_kernels(
        [[kernel_description], caption], sample_x, sample_y, y_std
    )

    return {
        "ground_truth": {
            "kernel": kernel_description,
            "lml": train_lml[0],
            "train_rmse": train_rmse[0],
            "train_lpd": train_lpd[0],
            "test_rmse": test_rmse[0],
            "test_lpd": test_lpd[0]
        },
        "prediction": {
            "kernel": " + ".join(caption),
            "lml": train_lml[1],
            "train_rmse": train_rmse[1],
            "train_lpd": train_lpd[1],
            "test_rmse": test_rmse[1],
            "test_lpd": test_lpd[1]
        }
    }


def main(args: Namespace):
    model = load_kitt(
        args.model_load_path,
        args.kitt_encoder,
        args.kitt_decoder,
        args.max_products,
        args.max_expression_length
    )

    x_values = load_random_coords(args.resolution, args.num_dimensions, args.x_scaling)
    sample_generator = SampleGenerator(
        x_values,
        args.resolution,
        batch_size=1,
        min_expression=1,
        max_expression=args.max_expression_length,
        sigma=1.0,
        use_fft=False,
        add_bias=False,
        standardise=True,
        all_dims=True,
        tokenizer=model.tokenizer,
        make_captions=True,
        include_x_with_samples=True
    )
    results = {
        kernel: single_kernel_evaluation(kernel, sample_generator, model)
        for kernel in model.tokenizer.kernel_tokens
    }
    with open(str(Path(args.log_dir) / "kitt_output_to_gpr_beyond_gtr.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    arguments = get_arguments()
    np.random.seed(arguments.seed)
    tf.random.set_seed(arguments.seed)
    main(arguments)
