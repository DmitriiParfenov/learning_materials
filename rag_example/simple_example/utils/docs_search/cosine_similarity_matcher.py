from typing import Callable

from rag_example.simple_example.protocols import TextPreprocessorProtocol


def find_best_match_by_cosine_similarity(
        query: str,
        db_corpus: list[str],
        metrics_calculator: Callable[[list[str], list[str], TextPreprocessorProtocol], float],
        text_cleaner: TextPreprocessorProtocol
) -> tuple[int, str]:
    """
    Функция последовательно перебирает все документы в `db_corpus`, вычисляет меру косинусное сходства между запросом и
    каждым документом с помощью функции `metrics_calculator` и возвращает документ с максимальным значением.
    Args:
        query: документ
        db_corpus: коллекция документов
        metrics_calculator: функция для расчета косинусного сходства
        text_cleaner: класс для предобработки текста
    Returns:
        tuple[int, str]
    """
    best_score, best_record = 0, ""
    for record in db_corpus:
        cosine_metric = metrics_calculator(query.split(), record.split(), text_cleaner)
        if cosine_metric > best_score:
            best_score, best_record = cosine_metric, record

    return best_score, best_record
