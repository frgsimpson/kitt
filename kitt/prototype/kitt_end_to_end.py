from typing import Optional, Union

import tensorflow as tf
import numpy as np

from kitt.data.tokeniser import KernelTokenizer
from kitt.networks.transformer.classification_transformer import ClassificationTransformer
from kitt.networks.transformer.transformer_decoder import TransformerDecoder

from kitt.prototype.evaluate_captions import (
    get_caption_indices_for_sample_from_transformer_decoder,
    get_caption_for_sample
)


class KITT(tf.keras.Model):
    def __init__(
            self,
            encoder: ClassificationTransformer,
            decoder: TransformerDecoder,
            tokenizer: KernelTokenizer,
            max_expression_length: int,
            min_expression_length: int = 1,
            return_text_captions: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.min_expression_length = min_expression_length
        self.max_expression_length = max_expression_length
        self.return_text_captions = return_text_captions
        self.run_eagerly = True

    def call(
        self,
        x: tf.Tensor,
        max_expression_length_override: Optional[int] = None,
        temperature: float = 1.0
    ):
        max_expression_length = max_expression_length_override or self.max_expression_length
        features = self.encoder.get_representations(x)

        return get_caption_indices_for_sample_from_transformer_decoder(
            features,
            self.decoder,
            self.tokenizer,
            tf.constant(max_expression_length, dtype=tf.int32),
            tf.constant(temperature)
        )

    def get_kernel_expression_for_sample(
            self,
            sample: Union[tf.Tensor, np.ndarray],
            temperature: float = 1.0,
            process_caption: bool = False,
            max_expression_length_override: Optional[int] = None
    ):
        max_expression_length = max_expression_length_override or self.max_expression_length
        return get_caption_for_sample(
            sample,
            self.tokenizer,
            max_expression_length,
            self.encoder,
            self.decoder,
            temperature,
            process_caption
        )



