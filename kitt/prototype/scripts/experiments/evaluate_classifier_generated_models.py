import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from copy import deepcopy

import numpy as np
import tensorflow as tf

from kitt.config import BASE_PATH
from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import make_kernel_from_label, load_random_coords
from kitt.data.tokeniser import KernelTokenizer
from kitt.prototype.evaluate import evaluate_prediction_vs_ground_truth
from kitt.utils.save_load_models import load_model


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
        "--classifier_load_path",
        default=(
            BASE_PATH / "saves" / "models" / "encoders" / "classifier-transformer"
        ),
        action="store_true",
        help="Do not save any log files to disk.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Directory where to save model and any relevant logs",
        default=Path(__file__).parent.parent.parent.parent.parent / "results",
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
        classifier: tf.keras.Model
):
    kernel = make_kernel_from_label(
        [kernel_description],
        sample_generator.ndims,
        sample_generator.sigma,
        all_dims=sample_generator.all_dims,
        add_bias=sample_generator.add_bias
    )
    example_data = sample_generator.make_batch_of_samples(kernel)
    x, y = example_data[0, ..., :-1], example_data[0, ..., -1:]
    return evaluate_prediction_vs_ground_truth(
        classifier,
        sample_generator.tokenizer,
        x,
        y,
        np.std(y),
        kernel_description,
        sample_generator.add_bias
    )


def main(args: Namespace):
    kernels = get_unique_product_kernels(max_terms=args.max_products)
    classifier = load_model(args.classifier_load_path)
    x_values = load_random_coords(args.resolution, args.num_dimensions, args.x_scaling)
    sample_generator = SampleGenerator(
        x_values,
        args.resolution,
        batch_size=1,
        min_expression=1,
        max_expression=1,
        sigma=1.0,
        use_fft=False,
        add_bias=False,
        standardise=True,
        all_dims=True,
        tokenizer=KernelTokenizer(vocabulary=deepcopy(kernels)),
        make_captions=False,
        include_x_with_samples=True
    )
    results = {
        kernel: single_kernel_evaluation(kernel, sample_generator, classifier) for kernel in kernels
    }
    with open(str(Path(args.log_dir) / "classifier_output_to_gpr_beyond_gtr.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    args = get_arguments()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)
