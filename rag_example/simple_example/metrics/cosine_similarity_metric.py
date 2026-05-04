from collections import Counter

import numpy as np

from rag_example.simple_example.protocols import TextPreprocessorProtocol


def calculate_cosine_similarity(
        text_1: list[str],
        text_2: list[str],
        text_cleaner: TextPreprocessorProtocol
) -> float:
    # Очищаем тексты.
    processed_text_1 = text_cleaner.normalize_text(text_1)
    processed_text_2 = text_cleaner.normalize_text(text_2)

    # Считаем частоту слов.
    vocabulary = sorted(set(processed_text_1 + processed_text_2))
    freq_1 = Counter(processed_text_1)
    freq_2 = Counter(processed_text_2)
    vector_1 = np.array([freq_1[word] for word in vocabulary])
    vector_2 = np.array([freq_2[word] for word in vocabulary])

    return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
