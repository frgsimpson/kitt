from argparse import ArgumentParser, Namespace

from kitt.config import DATASET_DIR


def parse_command_line_args() -> Namespace:
    """ Gather user overrides to the data generation config from the command line. """
    parser = ArgumentParser()

    parser.add_argument(
        "--base_config",
        type=str,
        help="Config from configs.py to use as a base from which arguments may be overridden.",
        default="regression"
    )
    parser.add_argument(
        "--num_dimensions",
        type=int,
        help="Number of input dimensions"
    )
    parser.add_argument(
        "--min_expression",
        type=int,
        help="Minimum complexity of the kernel expression",
    )
    parser.add_argument(
        "--max_expression",
        type=int,
        help="Maximum complexity of the kernel expression",
    )
    parser.add_argument(
        "--train_sigma",
        type=float,
        help="Breadth of priors in training set samples in log space",
    )
    parser.add_argument(
        "--test_sigma",
        type=float,
        help="Breadth of priors in test set samples in log space",
    )
    parser.add_argument(
        "--num_expressions",
        type=int,
        help="The number of random (kernel) expressions to generate samples from",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        help="Percentage of the whole dataset to assign to the training set",
    )
    parser.add_argument(
        "--num_samples_per_expression",
        type=int,
        help="The number of samples drawn from each kernel expression",
    )
    parser.add_argument(
        "--num_hyper_randomisations",
        type=int,
        help="The number of random hyperparameter randomisations per expression",
    )
    parser.add_argument(
        "--make_captions",
        dest="make_captions",
        action="store_true"
    )
    parser.add_argument(
        "--no_captions",
        dest="make_captions",
        action="store_false"
    )
    parser.add_argument(
        "--include_x_with_samples",
        dest="include_x_with_samples",
        action="store_true"
    )
    parser.add_argument(
        "--samples_without_x",
        dest="include_x_with_samples",
        action="store_false"
    )
    parser.add_argument(
        "--max_products",
        type=int,
        help="Maximum number of terms in product kernel."
        "Strongly influences size of the vocabulary.",
    )
    parser.add_argument(
        "--use_fft",
        dest="use_fft",
        action="store_true",
    )
    parser.add_argument(
        "--no_fft",
        dest="use_fft",
        action="store_false"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Name of the directory where to store files, relative to {DATASET_DIR}",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
    )
    parser.set_defaults(use_fft=None, include_x_with_samples=None, make_captions=None)
    args = parser.parse_args()
    return args
