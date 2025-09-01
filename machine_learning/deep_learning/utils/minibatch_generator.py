from typing import Generator

import numpy as np


def get_minibatch(
        x: np.ndarray,
        y: np.ndarray,
        size: int,
        shuffle: bool = True
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Функция для генерации мини-пакетов."""

    indexes = np.arange(x.shape[0])  # получили массив от 0 до len(x).
    if shuffle:
        np.random.shuffle(indexes)

    for idx in range(0, len(x) - 1, size):
        yield x[idx: idx + size], y[idx: idx + size]
