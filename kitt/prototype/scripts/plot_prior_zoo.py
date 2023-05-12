""" Create a set of samples from our vocabulary, to help identify if our priors are sensible. """
from pathlib import Path

import matplotlib.pyplot as plt

from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import (
    load_default_coords,
    order_sum_kernel_by_variance,
    randomise_hyperparameters,
    make_kernel_from_label,
)
from kitt.data.tokeniser import KernelTokenizer

MAX_PRODUCTS = 3  # Max length of a product kernel
SAVES_DIR = Path(__file__).parent.parent / "plots"

vocab = get_unique_product_kernels(max_terms=MAX_PRODUCTS)
tokenizer = KernelTokenizer(vocabulary=vocab)
resolution = 32

coords = load_default_coords(x_resolution=resolution, y_resolution=resolution)

generator = SampleGenerator(
    x_values=coords,
    x_resolution=resolution,
    batch_size=1,
    min_expression=1,
    max_expression=1,
    tokenizer=tokenizer,
    all_dims=True,  # try both
    make_captions=False,
    include_x_with_samples=False,
    iterations_per_epoch=1_000,
)


def get_sample(k_string):
    expression = [k_string]  # Full caption
    kernel = make_kernel_from_label(expression, 2, generator.sigma, generator.all_dims)
    randomise_hyperparameters(kernel)
    kernel = order_sum_kernel_by_variance(kernel)
    return generator.make_batch_of_samples(kernel)


nrows = 3
ncols = 4
# Systematically go through the vocabulary and sample batches from each kernel

for kernel_string in tokenizer.kernel_tokens:

    # Visualise
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5))

    for i in range(nrows):
        for j in range(ncols):
            image = get_sample(kernel_string)

            image = image.reshape(resolution, resolution)

            axs[i, j].imshow(image)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.suptitle(kernel_string)
    fig.subplots_adjust(hspace=0.2)

    # Save to file
    safe_kernel_string = kernel_string.replace('*', '_')
    filename = SAVES_DIR / safe_kernel_string
    plt.savefig(filename, resolution=200)
    plt.close(fig)
