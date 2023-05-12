from copy import deepcopy

import pytest
import numpy as np

from kitt.data.uci.dataset import Dataset, Boston


@pytest.mark.parametrize("dataset", [Boston])
def test_datasets_split(dataset: Dataset):
    ds_init_split_1 = dataset(split=1, prop=0.9)
    ds_split_2 = deepcopy(ds_init_split_1)
    ds_split_2.apply_split(2)
    ds_get_split_1 = deepcopy(ds_split_2)
    ds_get_split_1.apply_split(1)

    # Test that get split returns the right data
    assert np.allclose(ds_init_split_1.X_train, ds_get_split_1.X_train)
    assert np.allclose(ds_init_split_1.Y_train, ds_get_split_1.Y_train)
    assert np.allclose(ds_init_split_1.X_test, ds_get_split_1.X_test)
    assert np.allclose(ds_init_split_1.Y_test, ds_get_split_1.Y_test)

    # Test that get split is in fact having an effect
    assert not np.allclose(ds_init_split_1.X_train, ds_split_2.X_train)
    assert not np.allclose(ds_init_split_1.Y_train, ds_split_2.Y_train)
    assert not np.allclose(ds_init_split_1.X_test, ds_split_2.X_test)
    assert not np.allclose(ds_init_split_1.Y_test, ds_split_2.Y_test)
