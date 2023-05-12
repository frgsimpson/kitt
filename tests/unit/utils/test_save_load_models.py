from pathlib import Path

import numpy as np

from kitt.utils.save_load_models import load_model, instantiate_model, save_model

MODEL_CONSTRUCTION_KWARGS = {
    "network_identifier": "classifier-transformer",
    "n_classes": 34,
    "batch_size": 64,
    "hidden_units": 64,
    "attn_heads": 4,
    "num_input_dims": 2,
    "include_x_with_samples": True,
    "resolution": 64,
    "sample_shape": (64, 3),
    "irrelevant_kwarg": "to test for saving info in the same file"
}


def test_load_save_weights_does_not_smoke(tmpdir_factory):
    """
    Test the building, saving and loading of models and the consistency of such.
    """
    model_path_dir = Path(tmpdir_factory.mktemp("models"))
    model = instantiate_model(**MODEL_CONSTRUCTION_KWARGS)
    # The consistency is validated by checking that the weights are the same and that a forward pass
    # through the network yields the same outcome.
    # We generate random inputs for the forward pass.
    random_inputs = np.random.randn(1, 7, 2)
    model_out = model(random_inputs)
    model_reps = model.get_representations(random_inputs)
    save_model(model, model_path_dir, MODEL_CONSTRUCTION_KWARGS)
    loaded_model = load_model(model_path_dir)
    loaded_model_out = loaded_model(random_inputs)
    loaded_model_reps = loaded_model.get_representations(random_inputs)
    assert all([np.allclose(w1.numpy(), w2.numpy()) for w1, w2
                in zip(model.weights, loaded_model.weights)])
    assert np.allclose(model_out, loaded_model_out)
    assert np.allclose(model_reps, loaded_model_reps)
