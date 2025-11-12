from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class IMDBCustomCollate:
    def __init__(self, vocabulary: dict[str, int], tokenizer: Callable[[str], list[str]], device: str = 'cpu') -> None:
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.device = device

    def text_pipeline(self, text: str) -> list[int]:
        return [self.vocabulary[token] for token in self.tokenizer(text)]

    def __call__(self, batch: list[tuple[str, int]]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Преобразует батч из DataLoader в формат, подходящий для подачи в модель.
        Каждый элемент X (текстовая последовательность) сначала токенизируется и преобразуется в числовые индексы
        согласно словарю. Так как длины последовательностей различаются, для выравнивания используется
        nn.utils.rnn.pad_sequence, который дополняет более короткие последовательности нулями до длины самой длинной
        в батче.

        Args:
            batch: минипакет с X (последовательность) и Y (метка)
        Returns:
            tokens_list: тензор формата (batch_size * sequence_length)
            labels_list: тензор с метками формата (batch_size * 1)
            labels_list: тензор длинами последовательностей ДО дополнения нулями формата (batch_size * 1)
        """
        tokens_list, labels_list, lengths_list = [], [], []
        for x_batch, y_batch in batch:
            tokens = torch.tensor(self.text_pipeline(x_batch), dtype=torch.int64)
            tokens_list.append(tokens)
            labels_list.append(y_batch)
            lengths_list.append(tokens.size(0))
        labels_list = torch.tensor(labels_list)
        lengths_list = torch.tensor(lengths_list)
        padded_token_list = nn.utils.rnn.pad_sequence(tokens_list, batch_first=True)
        return padded_token_list.to(self.device), labels_list.to(self.device), lengths_list.to(self.device)
