import argparse
from datetime import datetime

from kitt.prototype.scripts.experiments.run_on_uci import main as run_kitt_on_uci
from kitt.prototype.scripts.experiments.run_multi_train_parallel import main as multi_train_parallel
from kitt.prototype.scripts.experiments.aggregate_uci import main as aggregate_results

USE_DECODER = False

def get_unified_args():
    """ Parse and validate script arguments """
    parser = argparse.ArgumentParser()

    # From run_on_uci
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--test_train_split", type=float, default=0.9)
    parser.add_argument("--encoder_name", type=str, default="classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs")  # "kitt_encoder_classifier-transformer_209765_params")
    parser.add_argument("--decoder_name", type=str, default=None)   # "decoder-transformer_3780645_params")
    parser.add_argument("--max_terms", type=int, default=2, help="For the moment must match max_terms used in training.")
    parser.add_argument("--num_models_per_dataset", type=int, default=3)
    parser.add_argument("--date_str", type=str, default=None)
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.add_argument("--full_kitt", dest="full_kitt", action="store_true")
    parser.add_argument("--classifier_only", dest="full_kitt", action="store_false")

    # From multi_train_parallel)
    parser.add_argument("--max_iters", type=int, default=10_000)
    parser.add_argument("--n_jobs", type=int, default=1)

    # From Aggregate results
    parser.add_argument(
        "--results_suffix",
        type=str,
        help="Suffix of results files (excluding extension), e.g. 'rbf_results' or 'multi_train_results'",
        default='multi_train_results'
    )

    parser.set_defaults(subset=True, full_kitt=USE_DECODER)

    parsed_args = parser.parse_args()
    date = datetime.now().strftime('%b%d')
    parsed_args.date_str = date
    parsed_args.dates=[date]
    parsed_args.date=[date]

    parsed_args.rbf_baseline_experiment = False

    return parsed_args


def main():
    args = get_unified_args()
    print("\n\n\nRunning Trained KITT Model on UCI")
    run_kitt_on_uci(args)
    print("\n\n\nRunning KITT Predicted Kernels on UCI Datasets")
    multi_train_parallel(args)
    print("\n\n\nCollecting Results Accross Datasets")
    aggregate_results(args)


if __name__ == "__main__":
    main()
