""" Wonder if the vocab in the Tokenizer looks sensible? You've come to the right place. """
import argparse

from kitt.data.kernels import get_unique_product_kernels
from kitt.data.tokeniser import KernelTokenizer

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--max_products", help="Max number of terms in the product kernel", default=3)
args = parser.parse_args()

vocab = get_unique_product_kernels(max_terms=args.max_products)
tokenizer = KernelTokenizer(vocabulary=vocab)

for kernel, i in tokenizer.word_index.items():
    if not kernel[0] == '<':  # Omit start, end, pad tokens
        print(kernel)
