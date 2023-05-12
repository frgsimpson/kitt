""" Test :mod kitt.utils.misc """
import argparse
from unittest.mock import patch

from kitt.utils.misc import get_args_string, yes_or_no


def test_get_args_string() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_train_steps", type=int, help="Number of training steps", default=10_000
    )
    parser.add_argument("--num_test_steps", type=int, help="Number of test steps", default=100)
    parsed_args = parser.parse_args(["--num_train_steps", "10", "--num_test_steps", "42"])

    assert get_args_string(parsed_args) == " num_test_steps: 42\nnum_train_steps: 10\n"


def test_yes_or_no__yes():
    with patch("builtins.input") as mock_input:
        mock_input.return_value = "yES!"
        assert yes_or_no("Lorem Ipsum?")

    with patch("builtins.input") as mock_input:
        mock_input.return_value = "Yessss"
        assert yes_or_no("Lorem Ipsum?")


def test_yes_or_no__no():
    with patch("builtins.input") as mock_input:
        mock_input.return_value = "nOOO!"
        assert not yes_or_no("Lorem Ipsum?")

    with patch("builtins.input") as mock_input:
        mock_input.return_value = "Noooo"
        assert not yes_or_no("Lorem Ipsum?")
