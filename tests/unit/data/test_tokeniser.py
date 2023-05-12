""" Tests for :mod kitt.data.tokeniser """

import numpy as np
import pytest

from kitt.data.tokeniser import PAD_TOKEN, KernelTokenizer

TOKENISER = KernelTokenizer()


def test_n_vocabulary():
    assert TOKENISER.n_vocabulary == 11


def test_pad_token_has_zero_index():
    assert not TOKENISER.word_index[PAD_TOKEN]
    assert TOKENISER.index_word[0] == PAD_TOKEN


def test_encode__pad():
    expression = ["PERIODIC", "RBF", "LINEAR"]
    encoding = TOKENISER.encode(expression=expression, pad=True, max_complexity=10)
    start = len(TOKENISER.kernel_tokens) + 1
    end = start + 1
    np.testing.assert_array_equal(encoding, np.array([[start, 7, 3, 2, end, 0, 0, 0, 0, 0, 0, 0]]))


def test_encode__no_pad():
    expression = ["PERIODIC", "RBF", "LINEAR"]
    encoding = TOKENISER.encode(expression=expression, pad=False, max_complexity=10)
    start = len(TOKENISER.kernel_tokens) + 1
    end = start + 1
    np.testing.assert_array_equal(encoding, np.array([[start, 7, 3, 2, end]]))


@pytest.mark.parametrize("pad", (True, False))
def test_encode_decode(pad: bool):
    expression = ["PERIODIC", "RBF", "LINEAR"]
    encoding = TOKENISER.encode(expression=expression, pad=pad, max_complexity=10)
    decoded = TOKENISER.decode(encoding)[0].split()
    # Remove the <start>, <end> and <pad> tokens.
    decoded_expression = [token for token in decoded if token not in {"<start>", "<end>", "<pad>"}]
    assert decoded_expression == expression
