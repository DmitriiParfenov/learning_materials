import re
from collections import Counter
from typing import Callable

from torch.utils.data.dataset import Subset


class VocabularyDict(dict):
    def __init__(self, unk_idx: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unk_idx = unk_idx

    def __getitem__(self, key):
        return super().get(key, self.unk_idx)


class SimpleTextVectorizer:
    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self._tokenizer = tokenizer or self._default_simple_tokenizer
        self._unk, self._unk_idx = '<unk>', 1
        self._pad, self._pad_idx = '<pad>', 0
        self._vocabulary = VocabularyDict(self._unk_idx)

    @staticmethod
    def _default_simple_tokenizer(text: str) -> list[str]:
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('[\W]+', ' ', text.lower())
        return text.split()

    def fit(self, dataset: Subset) -> None:
        total_tokens = Counter()
        for raw_text, label in dataset:
            tokens = self._tokenizer(raw_text)
            total_tokens.update(tokens)

        self._vocabulary[self._pad] = self._pad_idx
        self._vocabulary[self._unk] = self._unk_idx
        for index, token in enumerate(total_tokens, 2):
            self._vocabulary[token] = index

    @property
    def unknown_token(self):
        return self._unk

    @property
    def padding_token(self):
        return self._pad

    def get_tokenizer(self, text: str):
        return self._tokenizer(text)

    def get_vocabulary(self):
        assert self._vocabulary, 'Vocabulary is empty.'
        return self._vocabulary
