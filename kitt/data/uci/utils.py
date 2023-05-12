from copy import deepcopy

import numpy as np

from kitt.utils.training import Dataset

DATASETS_UCI = ["Boston", "WineRed", "Power", "Energy", "Concrete", "Kin8mn", "Yacht", "Naval"]

COMPARISON_DATASETS = [
        "Concrete",
        "Energy",
        "Boston",
        "Kin8mn",
        "Naval",
        "Power",
        "WineRed",
        "Yacht"
    ]


class ExperimentName:
    def __init__(self, base):
        self.s = base

    def add(self, name, value):
        self.s += f"_{name}-{value}"
        return self

    def get(self):
        return self.s


def subsample_dataset(dataset: Dataset, n_max: int, train_prop: float, seed: int = 0) -> Dataset:
    """
    Subsample a Dataset from the Bayesian Benchmarks library.
    Typically used to limit the number of data points.
    Note: This function has side effects mutating the dataset object passed in.

    :param dataset: The dataset to be subsampled.
    :param n_max: The maximal permitted number of points. If this exceeds the number of points in
        the dataset then the dataset is returned as is.
    :param train_prop: The proportion of the subsampled data to use for training (the remainder to
        be used for testing).
    :param seed: Seed for the random state used in subsampling (to ensure reproducibility).
    :return: A dataset with train and test data of at most n points.
    """
    # Only subsample where required
    if dataset.N <= n_max:
        return dataset

    # Deepcopy to avoid side effects
    ds = deepcopy(dataset)
    del dataset

    rs = np.random.RandomState(seed)
    # Collect all x and all y to resample as one
    x = np.vstack([ds.X_train, ds.X_test])
    y = np.vstack([ds.Y_train, ds.Y_test])

    # Sample indices then split up for test and train.
    indices = rs.choice(ds.N, n_max, replace=False)
    n_train = int(train_prop * n_max)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Update the properties of the dataset with the subsample information.
    ds.X_train = x[train_indices]
    ds.Y_train = y[train_indices]
    ds.X_test = x[test_indices]
    ds.Y_test = y[test_indices]
    ds.N = n_max

    return ds
