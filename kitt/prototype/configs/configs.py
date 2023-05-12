""" Some predefined configs for data generation. """

DEFAULT_RANDOM_SEED = 42

DATASET_GENERATION_CONFIGS = [
    "full-kitt",
    # "pattern",
    # "regression",
    # "simple-classification-transformer"
]


def load_config(name: str):
    """ Default dataset config for regression task. """

    config = {
        "num_dimensions": 4,
        "min_expression": 1,
        "max_expression": 1,
        "train_sigma": 1.,
        "test_sigma": 1.,
        "num_expressions": 200_000,
        "train_ratio": 0.9,
        "num_samples_per_expression": 1,
        "num_hyper_randomisations": 1,
        "make_captions": False,
        "include_x_with_samples": True,
        "max_products": 2,
        "use_fft": False,

        "random_seed": DEFAULT_RANDOM_SEED
    }

    if name == "pattern":
        overrides = {
            "num_dimensions": 2,
            "make_captions": True,
            "include_x_with_samples": False,
        }
    elif name == "regression":
        overrides = {
            "num_dimensions": 4,
            "make_captions": False,
            "include_x_with_samples": True,
        }
    elif name == "simple-classification-transformer":
        overrides = {
            "num_dimensions": 2,
            "make_captions": False,
            "include_x_with_samples": True,
            "max_products": 1,
        }
    elif name == "full-kitt":
        overrides = {
            "min_expression": 1,  # Otherwise overpredicts the end token
            "max_expression": 3,
            "max_products": 2,
            "make_captions": True,
            "num_hyper_randomisations": 1,
            "num_samples_per_expression": 1,
            "output_dir": name,
        }
    else:
        NotImplementedError("Unknown config requested", name)

    config.update(overrides)

    return config
