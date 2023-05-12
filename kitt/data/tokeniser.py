from typing import List

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from kitt.data.kernels import get_kernel_names

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
FILTERS = '!"#$%&()+.,-/:;=?@[\]^_`{|}~ '
PAD_INDEX = 0


class KernelTokenizer(Tokenizer):
    """ A tokenizer which will encode kernels from their names. """

    def __init__(self, vocabulary=None):
        super().__init__(filters=FILTERS, lower=False)
        if vocabulary is None:
            vocabulary = get_kernel_names()
        vocabulary.append(START_TOKEN)
        vocabulary.append(END_TOKEN)
        self.fit_on_texts(vocabulary)
        self.n_vocabulary = len(vocabulary) + 1
        self.word_index[PAD_TOKEN] = PAD_INDEX
        self.index_word[PAD_INDEX] = PAD_TOKEN
        self.kernel_tokens = self._get_kernel_tokens()

    def _get_kernel_tokens(self) -> List[str]:
        """ All tokens except for special tokens such as start/end/pad """

        kernel_tokens = []
        for k in self.index_word.values():
            if not k[0] in "<*":
                kernel_tokens.append(k)

        return kernel_tokens

    def encode(self, expression: List[str], pad: bool = False, max_complexity: int = 10):
        """ Create a label from a kernel expression """
        # text_to_sequences takes a list of strings
        full_expression = [" ".join([START_TOKEN] + expression + [END_TOKEN])]

        encodings = self.texts_to_sequences(full_expression)
        if pad:
            padded_encodings = pad_sequences(
                encodings, maxlen=max_complexity + 2, padding="post"
            )
            return padded_encodings

        return encodings

    def decode(self, labels):
        """ Create kernel expressions from a sequence of captions. """

        return self.sequences_to_texts(labels)
