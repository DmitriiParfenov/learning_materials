import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BagOfWords:
    """Класс для перевода текста в числовой вектор с использованием модели Bag of words."""

    def __init__(self) -> None:
        self.vectorizer: CountVectorizer | TfidfVectorizer | None = None
        self.vocabulary: dict[str, int] | None = None

    def _vectorize_by_tf(self, docs: np.array):
        """
        Метод преобразует текст в числовой вектор через tf(t, d). В результате получаем вектор с необработанными
        частотами терминов из сырых документов docs.
        """
        self.vectorizer = CountVectorizer()  # Использует дефолтный токенизатор - можно поменять.
        self.vectorizer.fit(docs)
        self.vocabulary = self.vectorizer.vocabulary_
        return self

    def _vectorize_by_tf_idf(self, docs: np.array):
        """
        Метод преобразует текст в числовой вектор через tf-idf(t, d). В результате получаем вектор с нормализованными
        частотами слов из сырых документов docs. Также этот метод используется для понижения веса часто встречающихся
        слов в векторах, которые не несут полезной информации.
        """
        self.vectorizer = TfidfVectorizer(
            use_idf=True,
            norm='l2',  # нормализация вектора через длину вектора.
            smooth_idf=True  # назначаем нулевой вес, если термин встречается во всех документах.
        )   # Использует дефолтный токенизатор - можно поменять.
        self.vectorizer.fit(docs)
        self.vocabulary = self.vectorizer.vocabulary_
        return self

    def fit(self, raw_documents: np.array, method: str = 'tf-idf'):
        """
        Метод для обучения модели Bag of words.
        Создается словарь, где ключи – это уникальные слова или токены для каждого слова в сыром документе raw_documents,
        а значения – индексы уникальных слов или токенов.
        """
        if method not in ('tf', 'tf-idf'):
            raise ValueError("Method must be 'tf' or 'tf-idf'")

        if method == 'tf':
            self._vectorize_by_tf(raw_documents)
            return self

        self._vectorize_by_tf_idf(raw_documents)
        return self

    def transform(self, raw_documents: np.array) -> list[float | int]:
        if not self.vectorizer or not self.vocabulary:
            raise ValueError("Vocabulary not fitted or provided")

        return self.vectorizer.transform(raw_documents).toarray()

    def get_vocabulary(self) -> dict[str, int]:
        if not self.vocabulary:
            raise ValueError("Vocabulary not fitted or provided")
        return self.vocabulary
