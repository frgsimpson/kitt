""" Script to train a classifier to identify GP kernels from their samples. """
import argparse
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.python.keras.metrics import SparseTopKCategoricalAccuracy

from kitt.config import AVAILABLE_GPU, DATASET_DIR, TF_VERSION, SAVES_DIR
from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import load_default_coords, load_random_coords
from kitt.data.tokeniser import KernelTokenizer
from kitt.prototype.classifier.callbacks import TimeHistory
from kitt.utils.misc import get_args_string
from kitt.utils.save_load_models import instantiate_model
from kitt.utils.training import Dataset
from kitt.utils.save_load_models import save_model as save_model_fn

DEBUG = False


def get_arguments() -> argparse.Namespace:
    """ Parse and validate standard experimental arguments """

    parser = argparse.ArgumentParser()

    n_dimensions = 4
    resolution = 64

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        help="Number of training steps in one epoch",
        default=1_000
    )
    parser.add_argument(
        "--num_dimensions",
        type=int,
        help="Dimensionality of the data",
        default=n_dimensions,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "--num_test_steps",
        type=int,
        help="Number of test steps",
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=128,
    )
    parser.add_argument(
        "--disposable_model",
        default=False,
        action="store_true",
        help="Do not save the trained model to disk.",
    )
    parser.add_argument(
        "--disposable_logs",
        default=False,
        action="store_true",
        help="Do not save any log files to disk.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Base directory from which to set up model saving and any relevant logging",
        default=SAVES_DIR,
    )
    parser.add_argument(
        "--network_name",
        default="classifier-transformer",
        help="Select network architecture, "
             "such as classifier-dense, classifier-shallow, classifier-transformer or resnext38_32x4d",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="The lengthscale spanned by data",
        default=5,
    )
    parser.add_argument(
        "--num_attn_heads",
        type=int,
        help="The number of attention heads used in the transformer(s).",
        default=4,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Data resolution",
        default=resolution,
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        help="The number of hidden units used in the neural nets"
        "(i.e. the internal representation dimension).",
        default=128,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="64_4d_max_exp_1_prod_2_random_inputs",
        help=f"Name of the dataset archive (with no extension), relative to {DATASET_DIR}. "
        "If None uses a SampleGenerator as per the above arguments.",
    )
    parsed_args = parser.parse_args()
    parsed_args.save_logs = not parsed_args.disposable_logs
    parsed_args.save_model = not parsed_args.disposable_model

    return parsed_args


def classifier_training(
    network_identifier: str,
    resolution: int,
    steps_per_epoch: int,
    num_train_epochs: int,
    num_test_steps: int,
    batch_size: int,
    num_dimensions: int,
    save_dir: Path,
    save_logs: bool,
    save_model: bool,
    dataset_name: Optional[str] = None,
    hidden_units: int = 32,
    attn_heads: int = 8,
):
    include_x_with_samples = network_identifier == "classifier-transformer"
    if save_logs:
        log_dir = save_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(str(log_dir / "arguments.txt"), "w") as f:
            f.write(get_args_string(args))

    if dataset_name:
        print("Loading data:", dataset_name)

        data = Dataset(dataset_name)
        train_data = data.get_tf_training_set().repeat().shuffle(10_000).batch(batch_size)
        test_data = data.get_tf_validation_set().repeat().shuffle(10_000).batch(batch_size)
        tokenizer = data.tokenizer
        features_spec, labels_spec = train_data.element_spec
        features_shape = tuple(features_spec.shape.as_list())
        labels_shape = tuple(labels_spec.shape.as_list())
    else:
        print("Generating data dynamically")
        if include_x_with_samples:
            coords = load_random_coords(n_samples=resolution, n_dims=num_dimensions)
        else:
            coords = load_default_coords(x_resolution=resolution, y_resolution=resolution)

        max_products = 2  # Max length of a product kernel
        vocab = get_unique_product_kernels(max_terms=max_products)
        tokenizer = KernelTokenizer(vocabulary=vocab)

        train_data = SampleGenerator(
            x_values=coords,
            x_resolution=resolution,
            batch_size=batch_size,
            min_expression=1,
            max_expression=1,
            tokenizer=tokenizer,
            make_captions=False,
            include_x_with_samples=include_x_with_samples,
            iterations_per_epoch=steps_per_epoch
        )
        # Data is generated on the fly so no need to worry about explicitly splitting our test data.
        test_data = train_data
        features, labels = train_data.make_batch()
        features_shape, labels_shape = features.shape, labels.shape

    class_names = tokenizer.kernel_tokens
    n_classes = len(class_names)
    sample_shape = features_shape[1:]

    model_construction_args = {
        "network_identifier": network_identifier,
        "n_classes": n_classes,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "attn_heads": attn_heads,
        "num_input_dims": num_dimensions,
        "include_x_with_samples": include_x_with_samples,
        "resolution": resolution,
        "sample_shape": sample_shape,
    }
    info = {
        "min_expression": 1,
        "max_expression": 1,
        "tokenizer_vocab": tokenizer.kernel_tokens,
        "make_captions": False,
        "include_x_with_samples": include_x_with_samples,
        "iterations_per_epoch": steps_per_epoch,
        "dataset": dataset_name
    }
    model = instantiate_model(**model_construction_args)

    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=50_000,
        decay_rate=0.1,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='Accuracy', dtype=tf.float64)
    topk_metric = SparseTopKCategoricalAccuracy(k=5, name='Top5', dtype=tf.float32)
    metrics = [accuracy_metric, topk_metric]

    model.compile(optimizer=optimizer,
                  # Stable training requires no softmax so from_logits=True
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics,
                  run_eagerly=DEBUG,
                  )

    print("Feature shape", features_shape)
    print("Labels shape", labels_shape)
    print("N classes", n_classes)

    time_callback = TimeHistory()
    model.fit(
        x=train_data,
        epochs=num_train_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[time_callback],
    )
    times = time_callback.times

    print("Training complete in", times)

    if save_model:
        model_parent_folder = network_identifier + f"_{dataset_name}" 
        model_dir = save_dir / "models" / "encoders" / model_parent_folder
        model_dir.mkdir(parents=True, exist_ok=True)
        info.update(model_construction_args)
        save_model_fn(model, model_dir, model_construction_args)

    test_loss, test_acc, test_top5 = model.evaluate(test_data, verbose=2, steps=num_test_steps)

    return test_loss, test_acc, test_top5


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float64")
    args = get_arguments()
    
    print(f"Tensorflow v{TF_VERSION}")
    print(f"GPUs: {AVAILABLE_GPU}")

    test_loss, test_acc, test_top5 = classifier_training(
        args.network_name,
        args.resolution,
        args.steps_per_epoch,
        args.num_train_epochs,
        args.num_test_steps,
        args.batch_size,
        args.num_dimensions,
        args.save_dir,
        args.save_logs,
        args.save_model,
        args.dataset,
        args.hidden_units,
        args.num_attn_heads,
    )

    print("\nTest loss:", test_loss)
    print("\nTest accuracy:", test_acc)
    print("\nTest Top 5 accuracy:", test_top5)
