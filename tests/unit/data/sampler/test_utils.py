""" Test for :mod kitt.data.sampler.utils """
from gpflow.kernels import RBF, Sum

from kitt.data.sampler.utils import order_sum_kernel_by_variance


def test_order_sum_kernel_by_variance():

    kernel_a = RBF(variance=2.0)
    kernel_b = RBF(variance=10.0)

    sum_kernel = Sum([kernel_a, kernel_b])

    ordered_kernel = order_sum_kernel_by_variance(sum_kernel)

    assert ordered_kernel.kernels[0].variance.numpy() == 10.0
    assert ordered_kernel.kernels[1].variance.numpy() == 2.0
