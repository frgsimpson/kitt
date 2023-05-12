import numpy as np
import pytest

from kitt.data.kitt_kernels import ShiftedLinear


@pytest.mark.parametrize("variances", ([1., 2.], [0.8, 0.9, 1.0]))
def test_shifted_linear(variances):
    """
    Ordinary linear kernels have vanishing variance at the origin.
    The ShiftedLinear enables this vanishing point to move around arbitrarily.
    This is particularly important when using Linear*Noise as it enables a downward sloping noise gradient.
    Variance automatically scales with shifting such that near the origin the asymptotic value of K(0,0)
    is the variance parameter.
    """

    n_dims = len(variances)
    active_dims = range(n_dims)

    shift_size = 1_000.
    shift_vector = shift_size * np.ones(n_dims)
    new_origin = -1 * shift_vector[None, :]
    kernel = ShiftedLinear(active_dims=active_dims, variance=variances)
    kernel.shifts.assign(shift_vector)

    shifted_origin_variance = kernel(new_origin, new_origin).numpy()
    origin_variance = kernel(0. * new_origin, 0. * new_origin).numpy()
    expected_origin_variance = np.sum(variances)

    assert shifted_origin_variance == 0., 'Variance not vanishing at origin'
    assert np.isclose(origin_variance, expected_origin_variance, rtol=1e-3), 'Unexpected variance in linear kernel'

