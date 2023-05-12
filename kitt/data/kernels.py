""" A collection of gpflow kernels and their corresponding tokens. """
import random
import re
from enum import Enum, unique
from itertools import combinations_with_replacement
from typing import List, Sequence

import gpflow as gf
import numpy as np

from kitt.data.kitt_kernels import ShiftedLinear, Periodic, Cosine


@unique
class KernelType(Enum):
    """
    A collection of kernel types we wish to identify.
    """

    NOISE = 1
    LINEAR = 2
    RBF = 3
    MATERN12 = 4
    MATERN32 = 5
    MATERN52 = 6
    PERIODIC = 7
    COSINE = 8


def get_kernel_names() -> List[str]:
    """ Collect a list of valid kernel names. """

    return [e.name for e in KernelType]


def get_unique_product_kernels(max_terms: int) -> List[str]:
    """ Collect all unique product kernels up to a maximum complexity of max_terms"""

    product_kernels = get_product_kernels(max_terms=max_terms)

    return remove_redundant_kernels(product_kernels)


def get_product_kernels(max_terms: int, primitive_kernels=None) -> List[str]:
    """Collect a list of valid kernel names, including products of primitive kernels.
    Product terms are restricted to a single permutation to avoid duplicates (A*B but not B*A)."""

    if primitive_kernels is None:
        primitive_kernels = get_kernel_names()
        primitive_kernels.remove("NOISE")  # Noise is not valid in a product with stationary kernels

    product_kernels = []
    for i in range(max_terms):
        for combination in combinations_with_replacement(primitive_kernels, i + 1):
            kernel = "*".join(combination)
            product_kernels.append(kernel)

    # Add valid noise kernels manually
    product_kernels.append("NOISE")
    if max_terms > 1:
        product_kernels.append("LINEAR*NOISE")

    return product_kernels


def remove_redundant_kernels(kernels) -> List[str]:
    """ Certain kernel combinations are redundant, such as rbf * rbf """

    def is_redundant(kernel: str) -> bool:

        duplicates = [
            "RBF",
            "PERIODIC",
            "COSINE"
        ]  # These kernels should only appear once in a product.

        for d in duplicates:
            entries = re.findall(d, kernel)
            if len(entries) > 1:
                return True

        return False

    unique_kernels = [k for k in kernels if not is_redundant(k)]

    return unique_kernels


def choose_active_dims(ndims: int) -> Sequence[int]:
    """ Select a random set of dimensions to be active. """

    # Decide total number of active dims
    n_active = np.random.randint(low=1, high=1 + ndims)

    # Now choose which dimensions they belong to
    return random.sample(range(ndims), n_active)


def get_kernel_map():
    return {
        KernelType.NOISE.name: (gf.kernels.White, dict()),
        KernelType.RBF.name: (gf.kernels.RBF, dict()),
        KernelType.MATERN12.name: (gf.kernels.Matern12, dict()),
        KernelType.MATERN32.name: (gf.kernels.Matern32, dict()),
        KernelType.MATERN52.name: (gf.kernels.Matern52, dict()),
        KernelType.LINEAR.name: (ShiftedLinear, dict()),
        KernelType.PERIODIC.name: (Periodic, dict(period=0.1)),
        KernelType.COSINE.name: (Cosine, dict()),
    }
