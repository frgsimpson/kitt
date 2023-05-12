""" Testing the :mod kitt.utils.training module """
import pytest
import tensorflow as tf

from kitt.data.tokeniser import KernelTokenizer
from kitt.utils.training import Dataset


@pytest.mark.parametrize(
    "dataset_name,input_shape",
    [
        ("test_no_channel", (32, 32)),
    ]
)
def test_dataset_attributes(dataset_name, input_shape):
    dataset = Dataset(dataset_name)

    assert dataset.num_expressions("train") == 8
    assert dataset.num_samples_per_expression("train") == 32
    assert dataset.num_expressions("test") == 2
    assert dataset.num_samples_per_expression("test") == 32
    assert dataset.ndims == 2
    assert dataset.x_resolution == 32
    assert dataset.y_resolution == 32
    assert dataset.num_datapoints == 1024
    assert dataset.min_expression == 2
    assert dataset.max_expression == 4
    assert dataset.use_fft
    assert dataset.num_train_instances == 256
    assert dataset.num_test_instances == 64
    assert isinstance(dataset.tokenizer, KernelTokenizer)
    assert dataset.input_shape == input_shape
    assert dataset.target_shape == (dataset.max_expression + 2,)


@pytest.mark.parametrize("dataset_name", ["test_no_channel"])
def test_dataset__get_tf_dataset(dataset_name):
    dataset = Dataset(dataset_name)

    def check_tf_dataset(tf_dataset):
        assert isinstance(tf_dataset, tf.data.Dataset)

        items = list(tf_dataset.as_numpy_iterator())
        assert len(items) == dataset.num_train_instances or len(items) == dataset.num_test_instances
        features, labels = items[0]
        assert features.shape == dataset.input_shape
        assert labels.shape == dataset.target_shape

    check_tf_dataset(dataset.get_tf_training_set())
    check_tf_dataset(dataset.get_tf_validation_set())
