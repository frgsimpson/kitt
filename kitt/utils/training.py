import pickle
import shutil
import tarfile
from typing import Optional

import h5py
import tensorflow as tf

from kitt.config import DATASET_DIR
from kitt.data.tokeniser import KernelTokenizer


class Dataset:
    """ An abstraction on top of a dataset archive """

    def __init__(
        self,
        name: str,
    ) -> None:
        """
        :param name: name of the file (under DATASET_DIR), '.tar.gz' suffix assumed
        """
        self.name = name
        self._file_path = DATASET_DIR / name
        try:
            with tarfile.open(self._file_path.with_suffix(".tar.gz"), "r:gz") as archive:
                archive.extractall(str(DATASET_DIR))
        except:
            pass

        self._train = h5py.File(self._file_path / "train.h5", "r")
        self._test = h5py.File(self._file_path / "test.h5", "r")
        assert self._train.attrs["min_expression"] == self._test.attrs["min_expression"]
        assert self._train.attrs["max_expression"] == self._test.attrs["max_expression"]
        assert self._train.attrs["num_dimensions"] == self._test.attrs["num_dimensions"]
        assert self._train.attrs["x_resolution"] == self._test.attrs["x_resolution"]
        assert self._train.attrs["y_resolution"] == self._test.attrs["y_resolution"]
        assert self._train.attrs["y_resolution"] == self._test.attrs["y_resolution"]
        assert self._train.attrs["fft"] == self._test.attrs["fft"]

        assert self._train["X"].shape[1:] == self._test["X"].shape[1:]
        assert self._train["Y"].shape[1:] == self._test["Y"].shape[1:]

        with open(self._file_path / "tokenizer.pickle", "rb") as f:
            self._tokenizer = pickle.load(f)

    def __del__(self):
        self._train.close()
        self._test.close()
        if self._file_path.exists():
            shutil.rmtree(str(self._file_path), ignore_errors=False)

    def _select_file(self, dataset: str):
        if dataset == "train":
            return self._train
        elif dataset == "test":
            return self._test

        raise ValueError("Unknown dataset")

    def num_expressions(self, dataset: str) -> int:
        return self._select_file(dataset).attrs["num_expressions"]

    def num_samples_per_expression(self, dataset: str) -> int:
        return self._select_file(dataset).attrs["num_samples_per_expression"]

    @property
    def ndims(self) -> int:
        return self._train.attrs["num_dimensions"]

    @property
    def x_resolution(self) -> int:
        return self._train.attrs["x_resolution"]

    @property
    def y_resolution(self) -> int:
        return self._train.attrs["y_resolution"]

    @property
    def num_datapoints(self) -> int:
        return self.x_resolution * self.y_resolution

    @property
    def min_expression(self) -> int:
        return self._train.attrs["min_expression"]

    @property
    def max_expression(self) -> int:
        return self._train.attrs["max_expression"]

    @property
    def use_fft(self) -> bool:
        return self._train.attrs["fft"]

    @property
    def tokenizer(self) -> KernelTokenizer:
        return self._tokenizer

    @property
    def num_train_instances(self) -> int:
        return self._train["X"].shape[0]

    @property
    def num_test_instances(self) -> int:
        return self._test["X"].shape[0]

    @property
    def input_shape(self):
        return self._train["X"].shape[1:]

    @property
    def target_shape(self):
        return self._train["Y"].shape[1:]

    def get_tf_training_set(self) -> tf.data.Dataset:
        return self._get_tf_dataset("train")

    def get_tf_validation_set(self) -> tf.data.Dataset:
        return self._get_tf_dataset("test")

    def _get_tf_dataset(self, dataset: str) -> tf.data.Dataset:
        file = self._select_file(dataset)

        class Generator:
            def __init__(self, file) -> None:
                self._file = file

            def __call__(self):
                for features, label in zip(file["X"], file["Y"]):
                    yield features, label

        return tf.data.Dataset.from_generator(
            Generator(file),
            (tf.float64, tf.int32),
            (tf.TensorShape(list(self.input_shape)), tf.TensorShape(list(self.target_shape))),
        )
