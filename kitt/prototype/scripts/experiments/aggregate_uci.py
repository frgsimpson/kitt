"""
Script to compute aggregated metrics (mean +- stddev) over experimental results
obtained over different splits across a number of (UCI) datasets.
"""
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import json
import numpy as np
import pandas as pd

from kitt.config import LOG_DIR
from kitt.data.uci import dataset as uci_datasets
import matplotlib.pyplot as plt


def parse_command_line_args() -> Namespace:
    """ As it says on the tin """
    parser = ArgumentParser()

    parser.add_argument(
        "--results_suffix",
        type=str,
        help="Suffix of results files (excluding extension), e.g. 'rbf_results' or 'multi_train_results'",
        default='multi_train_results'
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="*",
        help=f"Name of the log directories containing experimental results, format is e.g. Feb12."
             f"Specify more than one in case the experiment run overnight.",
        default=["May17"]
    )

    return parser.parse_args()


def get_dataset_class(dataset):
    return getattr(uci_datasets, dataset)


def standard_error(x):
    return np.std(x) / np.sqrt(len(x))


def grouped_bar_plot(labels, heights, ylabel, title,  yerr):
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, heights[0].values, width, label="Top 3 captions", color="b", alpha=0.7, yerr=yerr[0], capsize=5)
    ax.bar(x + width/2, heights[1].values, width, label="Top 5 captions", color="r", alpha=0.5, yerr=yerr[1], capsize=5)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


def main(args):
    # Read the results files
    data = []
    paths = []
    for date in args.dates:
        results_dir = LOG_DIR / date / 'results'
        results_files = list(results_dir.glob(f"*{args.results_suffix}.json"))
        paths.extend(results_files)
        for result_file in results_files:
            with open(result_file) as json_file:
                data.append(json.load(json_file))

    if len(paths) == 0:
        print("No results found - try a different results suffix")
        sys.exit(2)

    # Process the data
    df = pd.DataFrame.from_records(data)
    df["paths"] = paths
    df["N"] = [get_dataset_class(d).N for d in df.dataset_name]
    df["D"] = [get_dataset_class(d).D for d in df.dataset_name]
    
    # Aggregate the tables
    metrics = [col for col in list(df.columns) if (col.endswith("_rmse") or col.endswith("_nlpd"))]
    summary = df.groupby(["dataset_name"])[metrics].agg(["mean", standard_error])
    summary.to_csv(str(results_dir / "aggregated_results.csv"))
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(summary)

if __name__ == "__main__":
    
    args = parse_command_line_args()
    main(args)
