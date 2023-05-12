""" Smoke test classifier training"""
import pytest

from kitt.prototype.scripts.train_classifier import classifier_training


@pytest.mark.parametrize("network_name", ["classifier-dense", "resnext38_32x4d"])
def test_clasifier_training_does_not_smoke(network_name: str):
    try:
        classifier_training(
            network_name,
            resolution=28,
            steps_per_epoch=1,
            num_train_epochs=1,
            num_test_steps=1,
            batch_size=2,
            num_dimensions=2,
            save_dir=None,
            save_logs=False,
            save_model=False,
        )
    except:
        pytest.fail("Could not train classifier")


def test_transformer_clasifier_training_does_not_smoke():
    try:
        classifier_training(
            network_identifier="classifier-transformer",
            resolution=28,
            steps_per_epoch=1,
            num_train_epochs=1,
            num_test_steps=1,
            batch_size=2,
            num_dimensions=4,
            save_dir=None,
            save_logs=False,
            save_model=False,
        )
    except:
        pytest.fail("Could not train classifier")
