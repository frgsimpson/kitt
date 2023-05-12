from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf

from kitt.data.sampler.sample_generation import SampleGenerator
from kitt.data.sampler.utils import standardise_samples
from kitt.data.tokeniser import START_TOKEN, KernelTokenizer, END_TOKEN
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.rnn_decoder import RNN_Decoder
from kitt.networks.transformer.transformer_decoder import TransformerDecoder

N_EVALS = 60  # How many samples to draw from the probabilistic prediction
MIN_CAPTION_LENGTH = 1  # Can be used to enforce more complex captions


def infer_top_n_captions(encoder: ClassificationTransformer,
                         decoder: RNN_Decoder,
                         sample_generator,
                         x_train, y_train, n_captions: int):
    """
    Get the most probable captions based on the input data.
    The particular encoder/decoders used may change in future.
    Encoder takes the data as input. Decoder takes teh encoder's representations as input,
    along with a partially completed caption.
    Since inference is fast, we compute many possible captions and return the top n_captions
    favoured by the network.
    """

    # Preprocess in same manner as in training
    input_x = standardise_samples(x_train.T).T
    sample = np.concatenate((input_x, y_train), axis=-1)

    captions_and_weights = multi_evaluate(sample, sample_generator, decoder, encoder)

    captions, weights = zip(*captions_and_weights)

    return captions[:n_captions], weights[:n_captions]


def multi_evaluate(sample: np.ndarray,
                   sample_generator: SampleGenerator,
                   decoder: Union[RNN_Decoder, TransformerDecoder],
                   encoder: tf.keras.Model,
                   n_evals: int = N_EVALS,
                   process_caption: bool = False) -> List:
    """
     Build a dictionary of viable predictions and their associated probabilities.

     :param sample - the piece of data, of expected shape seq x (xdim+1)
     :param sample_generator - the sampler used to train the network
     :param decoder
     :param encoder
     :param n_evals - how many random paths to draw in order to identify the most probable kernels
     :param process_caption - whether to convert captions into readable form
     :returns List of tuples of captions and their corresponding probabilities, in descending order
    """

    # First ensure a strong prediction by using a low temperature (still no guarantee it's the mode)
    predictions = []  # Store list of expression / probability tuples
    stored_expressions = set()  # We'll use this to avoid storing duplicates

    max_predict, prob = get_caption_for_sample(
        sample,
        sample_generator.tokenizer,
        sample_generator.max_expression,
        encoder,
        decoder,
        temperature=1e-6,
        process_caption=process_caption
    )

    prediction = (max_predict, prob)
    predictions.append(prediction)
    hashable_expression = ''.join(max_predict)
    stored_expressions.add(hashable_expression)

    # Now try stochastic predictions
    for _ in range(n_evals):
        expression, prob = get_caption_for_sample(
            sample,
            sample_generator.tokenizer,
            sample_generator.max_expression,
            encoder,
            decoder,
            temperature=1.0,
            process_caption=process_caption
        )
        prediction = (expression, prob)
        hashable_expression = ''.join(expression)
        if hashable_expression not in stored_expressions:
            predictions.append(prediction)
            stored_expressions.add(hashable_expression)

    # Finally sort the list of predictions by their assigned probability
    predictions.sort(key=lambda tup: tup[1], reverse=True)
    return predictions


def get_caption_for_sample(
        sample: Union[tf.Tensor, np.ndarray],
        tokenizer: KernelTokenizer,
        max_expression_length: int, encoder: tf.keras.Model,
        decoder: Union[RNN_Decoder, TransformerDecoder],
        temperature=1.0,
        process_caption: bool = False
) -> Tuple[List, float]:
    """ Infer the kernel used to create a given sample. """

    if hasattr(encoder, "get_representations"):  # Omit final classification layer
        features = encoder.get_representations(sample[None, :, :])
    else:
        features = encoder(sample[None, :, :])

    if isinstance(decoder, RNN_Decoder):
        caption, expression_probability = get_caption_for_sample_from_rnn_decoder(
            features,
            decoder,
            tokenizer,
            max_expression_length,
            temperature
        )
    elif isinstance(decoder, TransformerDecoder):
        caption, expression_probability = get_caption_for_sample_from_transformer_decoder(
            features,
            decoder,
            tokenizer,
            max_expression_length,
            temperature
        )
    else:
        raise ValueError("Decoder is not of a supported type.")

    if process_caption:
        caption = process_kernel_expression(caption)

    return caption, expression_probability


@tf.function
def get_caption_indices_for_sample_from_transformer_decoder(
    features: tf.Tensor,
    decoder: TransformerDecoder,
    tokenizer: KernelTokenizer,
    max_expression_length: tf.Tensor,
    temperature: tf.Tensor = tf.constant(1.0)
) -> Tuple[tf.Tensor, float]:
    expression_probability = tf.constant(1.0, dtype=tf.float64)
    output = tokenizer.word_index[START_TOKEN] * tf.ones((1, max_expression_length), dtype=tf.int32)

    for i in range(max_expression_length):
        # Must be TransformerDecoder
        logits = decoder.get_logits_for_prompt(output[:i+1], features)

        if i < MIN_CAPTION_LENGTH:  # Exclude end token from consideration
            logits = logits[:, :-1]
        alt_logits = logits / tf.cast(temperature, logits.dtype)

        predicted_id = tf.random.categorical(alt_logits, 1)[0][0]
        word_probability = tf.nn.softmax(logits)[0, predicted_id]
        expression_probability *= word_probability
        if predicted_id == tokenizer.word_index[END_TOKEN]:
            break

        output = tf.concat(
            [output[:, :i+1], tf.cast(tf.expand_dims([predicted_id], 0), output.dtype), output[:, i+2:]],
            axis=-1
        )
    return output, expression_probability


def get_caption_for_sample_from_transformer_decoder(
        features: tf.Tensor,
        decoder: TransformerDecoder,
        tokenizer: KernelTokenizer,
        max_expression_length: int,
        temperature: float = 1.0
) -> Tuple[List[str], float]:
    caption_ids, expression_probability = get_caption_indices_for_sample_from_transformer_decoder(
        features,
        decoder,
        tokenizer,
        tf.constant(max_expression_length, dtype=tf.int32),
        tf.constant(temperature)
    )
    caption_np = caption_ids.numpy()
    start_index = tokenizer.word_index[START_TOKEN]
    # Remove all start tokens (which we also used for padding)
    caption_np = caption_np[caption_np != start_index].reshape((1, -1))
    caption = tokenizer.sequences_to_texts(caption_np)
    caption = caption[0].split()
    return caption, expression_probability.numpy()


def get_caption_for_sample_from_rnn_decoder(
    features: tf.Tensor,
    decoder: TransformerDecoder,
    tokenizer: KernelTokenizer,
    max_expression_length: int,
    temperature: float = 1.0
) -> Tuple[List[str], float]:
    output = tf.expand_dims([tokenizer.word_index[START_TOKEN]], 0)
    caption = []
    expression_probability = 1.0
    hidden_state = decoder.reset_state(batch_size=1)

    for i in range(max_expression_length):
        logits, hidden, _ = decoder(output, features, hidden_state)

        if i < MIN_CAPTION_LENGTH:  # Exclude end token from consideration
            logits = logits[:, :-1]
        alt_logits = logits / temperature

        predicted_id = tf.random.categorical(alt_logits, 1)[0][0].numpy()
        word_probability = tf.nn.softmax(logits)[0, predicted_id].numpy()
        expression_probability *= word_probability
        predicted_word = tokenizer.index_word[predicted_id]
        if predicted_word == END_TOKEN:
            break
        else:
            caption.append(predicted_word)

        output = tf.expand_dims([predicted_id], 0)

    return caption, expression_probability


def process_kernel_expression(expression: List[str]) -> str:
    """ Convert a caption expression into a readable form,
    by inserting addition operator. """

    result = " + ".join(expression)
    return result.replace("+ * +", "*")


def stochastic_sample(predictions, temperature: float = 1.0):
    """
    Helper function to sample an index from a probability array.
    At low temperature we pick the most probable element
    At high temperature we sample more freely.
    """

    if temperature <= 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature

    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)
