"""
Train and evaluate an captioning network to identify GP kernels from random samples.
Loads a pre-trained classification transformer which decomposes the data into features which
are then used to generate a full caption in a sequential fashion

Adapted from the image captioning and translation examples
https://www.tensorflow.org/tutorials/text/image_captioning
https://www.tensorflow.org/tutorials/text/transformer
"""

import argparse
from pathlib import Path
import time

import tensorflow as tf

from kitt.config import DATASET_DIR, SAVES_DIR
from kitt.data.kernels import get_unique_product_kernels
from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import load_random_coords
from kitt.data.tokeniser import KernelTokenizer, START_TOKEN
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.prototype.objectives_and_metrics import loss_function, accuracy_function
from kitt.utils.training import Dataset
from kitt.utils.save_load_models import save_model as save_model_fn, load_model
from kitt.networks.transformer.transformer_decoder import (
    TransformerDecoder,
    make_look_ahead_mask,
    make_padding_mask,
)
DTYPE = tf.float64


def get_arguments() -> argparse.Namespace:
    """ Parse and validate script arguments """
    parser = argparse.ArgumentParser()
    resolution = 256  # Sequence length - probably need more for captioning than for classifier
    n_dimensions = 4

    parser.add_argument(
        "--pretrained_encoder",
        default=None, # "classifier-transformer_64_4d_max_exp_1_prod_2_random_inputs",  # "classifier-transformer",
        type=str,
        help="Name of the network to load",
    )
    parser.add_argument(
        "--num_dimensions",
        type=int,
        help="Dimensionality of the data",
        default=n_dimensions,
    )
    parser.add_argument(
        "--max_caption_length",
        type=int,
        help="Maximal number of elements to include in the caption.",
        default=3
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        default=50,
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        help="Number of training steps in one epoch",
        default=1_000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1e-4
    )
    parser.add_argument(
        "--iters_per_update",
        type=int,
        help="Number of iterations per update",
        default=500
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Data resolution",
        default=resolution,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=128,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full-kitt", # "4d_max_exp_3_prod_2_captions_random_inputs",
        help=f"Name of the dataset archive (with no extension), relative to {DATASET_DIR}",
    )
    parser.add_argument(
        "--test_sigma",
        type=float,
        help="Breadth of priors in test set samples in log space",
        default=1.
    )
    parser.add_argument(
        '--plot',
        default=False,
        action='store_true',
        help="Plot evaluation"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Directory where to save model and any relevant logs",
        default=SAVES_DIR,
    )
    parser.add_argument(
        "--disposable_model",
        default=False,
        action="store_true",
        help="Do not save the trained model to disk.",
    )
    parser.add_argument(
        "--decoder_name",
        default="decoder-transformer",
        type=str,
        help="The name of the decoder model to use for KITT. "
             "Options: decoder-transformer, decoder-rnn"
    )
    parser.add_argument(
        "--pretrained_encoder_lr",
        default=None,
        type=float,
        help="The learning rate to train/fine tune the loaded encoder. "
             "If None, encoder not trained."
    )
    parsed_args = parser.parse_args()
    parsed_args.save_model = not parsed_args.disposable_model

    return parsed_args


class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, num_hidden_units, warmup_steps=4000):
        super(CustomLRSchedule, self).__init__()

        self.d_model = num_hidden_units
        self.d_model = tf.cast(self.d_model, DTYPE)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main() -> None:
    args = get_arguments()
    restore_checkpoint = False  # todo customise whether to attempt to restore from a previous train

    # Either train on the fly or use a stored dataset
    if args.dataset is not None:
        dataset = Dataset(args.dataset)
        dynamic_data = False

        n_train_samples = dataset.num_train_instances
        iters_per_epoch = n_train_samples // args.batch_size
        assert iters_per_epoch > 0, "Insufficient data to fill a single batch"
        # todo dataset.num_train_instances currently splits pure training data further into train/test

        training_set = dataset.get_tf_training_set()
        training_set = (
            training_set.shuffle(buffer_size=1_000)
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        # TODO: can use test data for validation during training
        validation_set = dataset.get_tf_validation_set()
        tokenizer = dataset.tokenizer

    else:
        print("Generating data dynamically  - ensure conditions match that used to train classifier")
        # Set up data generation in similar manner to train_classifier
        # EXCEPT now we make captions
        dynamic_data = True

        coords = load_random_coords(n_samples=args.resolution, n_dims=args.num_dimensions)
        max_products = 2  # Max length of a product kernel #
        # todo ensure max_products matches that used in train_classifier

        vocab = get_unique_product_kernels(max_terms=max_products)
        tokenizer = KernelTokenizer(vocabulary=vocab)

        training_set = SampleGenerator(
            x_values=coords,
            x_resolution=args.resolution,
            batch_size=args.batch_size,
            min_expression=3,  # Smallest caption length
            max_expression=args.max_caption_length,  # Longest caption length
            tokenizer=tokenizer,
            make_captions=True,
            include_x_with_samples=True,
            iterations_per_epoch=10_000
        )

    vocab_size = tokenizer.n_vocabulary

    # Start by loading the pretrained classification network
    if args.pretrained_encoder:
        # model_path = Path(__file__).parent.parent / "saves" / "encoders" / args.pretrained_encoder
        model_path = args.save_dir / "models/encoders" / args.pretrained_encoder
        encoder = load_model(model_path)
        print("Successfully loaded encoder. Only training decoder. ")
        if args.pretrained_encoder_lr:
            train_encoder = True
        else:
            train_encoder = False
    else:
        print("WARNING - Train from scratch - untested.")
        encoder = ClassificationTransformer(
            num_hidden_units=64,
            num_heads=4,
            num_classes=vocab_size,
        )
        encoder.compile()
        encoder.build((1, 2, 3))

        train_encoder = True

    if args.decoder_name.lower() == "decoder-rnn":
        # todo Experiment with embedding dim value
        decoder = RNN_Decoder(embedding_dim=8, units=128, vocab_size=vocab_size)
    elif args.decoder_name.lower() == "decoder-transformer":
        decoder = TransformerDecoder(
            num_units=256,
            num_heads=8,
            num_layers=6,
            vocab_size=vocab_size
        )
    else:
        raise NotImplementedError(f"Decoder of type {args.decoder_name} not supported.")
    model_path = "train_kitt_" + str(args.resolution)
    checkpoint_path = "kitt_checkpoints/" + model_path
    decoder_lr = CustomLRSchedule(decoder.num_units)
    decoder_optimizer = tf.keras.optimizers.Adam(1e-4)
    if train_encoder:
        encoder_optimizer = tf.keras.optimizers.Adam(args.pretrained_encoder_lr or args.lr)
        ckpt = tf.train.Checkpoint(
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer
        )
    else:
        ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=decoder_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    loss_plot = []
    if restore_checkpoint and ckpt_manager.latest_checkpoint:
        try:
            start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        except ValueError as ve:
            print(f"Load checkpoint failed. ValueError: {ve}.\nStarting from scratch.")
            start_epoch = 0
            # Reinitialise models to avoid corruption by partial restore.
            if args.decoder_name.lower() == "decoder-rnn":
                # todo Experiment with embedding dim value
                decoder = RNN_Decoder(embedding_dim=64, units=512, vocab_size=vocab_size)
            elif args.decoder_name.lower() == "decoder-transformer":
                decoder = TransformerDecoder(
                    num_units=512,
                    num_heads=8,
                    num_layers=6,
                    vocab_size=vocab_size
                )
            else:
                raise NotImplementedError(f"Decoder of type {args.decoder_name} not supported.")

            if train_encoder:
                encoder = ClassificationTransformer(
                    num_hidden_units=64,
                    num_heads=4,
                    num_classes=vocab_size,
                )
                encoder.compile()

    if args.decoder_name.lower() == "decoder-rnn":
        @tf.function
        def train_step(sample, target_sequence):

            loss = 0
            accuracy = 0
            hidden = decoder.reset_state(batch_size=target_sequence.shape[0])
            # Initial decoder input
            decoder_input = tf.expand_dims(
                [tokenizer.word_index[START_TOKEN]] * target_sequence.shape[0], 1
            )

            with tf.GradientTape(persistent=True) as tape:
                features = encoder.get_representations(sample)  # (n, d, k)

                for i in range(1, target.shape[1]):
                    predictions, hidden, _ = decoder(decoder_input, features, hidden)

                    loss += loss_function(target_sequence[:, i], predictions)
                    accuracy += accuracy_function(target_sequence[:, i], predictions)
                    # Using teacher forcing
                    decoder_input = tf.expand_dims(target_sequence[:, i], 1)

            n_predictions = int(target_sequence.shape[1]) - 1
            accuracy /= n_predictions
            average_loss = loss / n_predictions

            decoder_gradients = tape.gradient(loss, decoder.trainable_variables)
            decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
            if train_encoder:
                encoder_gradients = tape.gradient(loss, encoder.trainable_variables)
                encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
            del tape
            return loss, accuracy, average_loss

    elif args.decoder_name.lower() == "decoder-transformer":
        @tf.function
        def train_step(sample, target_sequence):
            target_output = target_sequence[:, 1:]
            decoder_prompts = target_sequence[:, :-1]
            look_ahead_mask = make_look_ahead_mask(tf.shape(decoder_prompts)[-1])
            padding_mask = make_padding_mask(decoder_prompts)
            # When processing the target sequence we want to mask both future values and padding.
            combined_mask = tf.maximum(padding_mask, look_ahead_mask)
            with tf.GradientTape(persistent=True) as tape:
                features = encoder.get_representations(sample, training=True)  # (n, d, k)

                predictions = decoder(
                    decoder_prompts,
                    features,
                    training=True,
                    look_ahead_mask=combined_mask
                )

                loss = loss_function(target_output, predictions)
            accuracy = accuracy_function(target_output, predictions)
            average_loss = loss  # Unlike RNN, here we have already averaged over the whole caption

            decoder_gradients = tape.gradient(loss, decoder.trainable_variables)
            decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
            if train_encoder:
                encoder_gradients = tape.gradient(loss, encoder.trainable_variables)
                encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
            del tape
            return loss, accuracy, average_loss
    else:
        raise NotImplementedError(f"Decoder of type {args.decoder_name} not supported.")

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        total_loss = 0
        train_time = 0
        n_epoch_iters = 0

        # todo wrap batching in tf function
        for (sample_tensor, target) in training_set:
            step_start = time.time()
            batch_loss, batch_accuracy, ave_loss = train_step(sample_tensor, target)
            train_time += time.time() - step_start
            total_loss += ave_loss.numpy()
            n_epoch_iters += 1

            if n_epoch_iters % args.iters_per_update == 0:
                print(
                    f"Epoch {epoch + 1} Batch {n_epoch_iters} "
                    f"Batch loss {ave_loss.numpy():.4f}, "
                    f"Batch accuracy {batch_accuracy.numpy():.4f}"
                )

            if dynamic_data and n_epoch_iters >= args.steps_per_epoch:
                break

        loss_plot.append(total_loss / n_epoch_iters)

        ckpt_manager.save()

        print("Epoch {} Loss {:.6f}".format(epoch + 1, total_loss / n_epoch_iters))
        print("Time taken for epoch {:.2f} sec\n".format(time.time() - start))
        print("Time taken for training steps {:.2f}\n".format(train_time))

    if args.save_model:
        decoder_construction_args = {
            "network_identifier": args.decoder_name,
            "decoder_name": args.decoder_name,
            "pretrained_encoder": args.pretrained_encoder,
            "n_classes": tokenizer.n_vocabulary,
            "batch_size": args.batch_size,
            "attn_heads": None,
            "num_input_dims": args.num_dimensions,
            "include_x_with_samples": True,
            "resolution": args.resolution,
            "sample_shape": None,
        }
        if args.decoder_name.lower() == "decoder-rnn":
            decoder_construction_args["hidden_units"] = decoder.embedding_dim
        elif args.decoder_name.lower() == "decoder-transformer":
            decoder_construction_args["hidden_units"] = decoder.num_units
            decoder_construction_args["attn_heads"] = decoder.num_heads
            decoder_construction_args["num_layers"] = decoder.num_layers
            decoder_construction_args["p_dropout"] = decoder.p_dropout
            decoder_construction_args["vocab_size"] = decoder.vocab_size
            decoder_construction_args["representation_dimensionality"] = encoder.num_hidden_units

        info = {
            "min_expression": 1,
            "max_expression": args.max_caption_length,
            "tokenizer_vocab": tokenizer.kernel_tokens,
            "make_captions": True,
            "include_x_with_samples": True,
            "iterations_per_epoch": args.steps_per_epoch,
            "dataset": args.dataset,
        }

        decoder_save_dir = (
                args.save_dir / "models" / "decoders" /
                f"{args.decoder_name}_{decoder.count_params()}_params"
        )
        decoder_save_dir.mkdir(parents=True, exist_ok=True)

        info.update(decoder_construction_args)
        save_model_fn(decoder, decoder_save_dir, info)

        if train_encoder:
            info["network_identifier"] = args.pretrained_encoder or "classifier-transformer"
            info["attn_heads"] = getattr(encoder, "num_heads", None)
            info["hidden_units"] = getattr(encoder, "num_hidden_units", None)

            encoder_save_dir = (
                    args.save_dir / "models" / "encoders" /
                    f"kitt_encoder_{info['network_identifier']}_{encoder.count_params()}_params"
            )
            encoder_save_dir.mkdir(parents=True, exist_ok=True)
            save_model_fn(encoder, encoder_save_dir, info)


if __name__ == "__main__":
    # tf.keras.backend.set_floatx('float32')
    main()
