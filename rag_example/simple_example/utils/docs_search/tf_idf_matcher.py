from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def setup_vectorizer(corpus: list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Функция преобразует каждый документ в корпусе corpus в численный вектор и возвращает объект векторизатора и
    разряженную матрицу TF-IDF.

    Матрица TF-IDF - это матрица, где строки - это документы, а столбцы - токены. Значение - это "важность" токена.
        coords (i, j): i - номер документа, j - номер токена в словаре
        values (float): "важность" токена. Если токен встречается в каждом предложении, то его вес уменьшается.
        IDF токена одинаковый в каждом предложении, но TF - нет. Следовательно, TF-IDF одного и того же токена
        в разных документах может отличаться.

    Пример:
                    человек   мужчина   работает   днем
        Документ 0   1.0      0         0        0
        Документ 1   0.385    0.652     0        0.652
        Документ 2   0.508    0         0.861    0
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def find_best_match_by_tf_idf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix: csr_matrix) -> tuple[int, int]:
    """
    Функция векторизует запрос в численный вектор, рассчитывает косинусное сходство между пользовательским вектором
    и всеми векторами в корпусе и возвращает индекс самого релевантного документа с максимальным значением.
    Args:
        query: str - документ
        vectorizer: TfidfVectorizer - объект обученного на документах из корпуса векторизатора
        tfidf_matrix: tfidf_matrix - tf-idf матрица
    Returns:
        tuple[int, int]
    """
    query_tfidf = vectorizer.transform([query])
    # Возвращает матрицу, где значения - это косинусное сходство между query и каждым документом из корпуса.
    cosine_metric = cosine_similarity(query_tfidf, tfidf_matrix)
    # Выбираем индекс самого релевантного документа.
    best_index = cosine_metric.argmax()
    best_score = cosine_metric[0, best_index]
    return best_score, best_index
