def find_best_match_by_keywords(query: str, db_corpus: list[str]) -> tuple[int, str]:
    """
    Сравнивает строку запроса `query` с набором документов `db_corpus` по совпадению ключевых слов и возвращает
    наиболее релевантный документ (по наибольшему количеству слов, пересекающихся в двух сравниваемых документах).
    Args:
        query: str - документ
        db_corpus: list[str] - коллекция документов
    Returns:
        tuple[int, str]
    """
    best_score = 0
    best_record = ""
    query_keywords = set(query.lower().strip().split())
    for doc in db_corpus:
        doc_keywords = set(doc.lower().strip().split())
        common_words_count = len(doc_keywords.intersection(query_keywords))
        if common_words_count > best_score:
            best_score, best_record = common_words_count, doc
    return best_score, best_record
