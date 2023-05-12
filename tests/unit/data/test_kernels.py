""" Tests for :mod kitt.data.kernels """
import random

import gpflow as gf
import pytest

from kitt.data.kernels import KernelType, get_kernel_map, get_kernel_names


@pytest.fixture
def kernel_map():
    return get_kernel_map()


def test_get_kernel_names():
    assert get_kernel_names() == [
        KernelType.NOISE.name,
        KernelType.LINEAR.name,
        KernelType.RBF.name,
        KernelType.MATERN12.name,
        KernelType.MATERN32.name,
        KernelType.MATERN52.name,
        KernelType.PERIODIC.name,
        KernelType.COSINE.name,
    ]


@pytest.mark.parametrize("kernel_name", get_kernel_names())
def test_get_kernel_map(kernel_map, kernel_name: str):

    assert isinstance(kernel_map, dict)

    kernel_params = kernel_map[kernel_name]
    assert isinstance(kernel_params[0](), gf.kernels.Kernel)
    assert isinstance(kernel_params[1], dict)
