"""
Простой RAG — это подход, при котором сначала осуществляется поиск наиболее релевантных документов из корпуса
(например, по совпадению ключевых слов), после чего найденные документы передаются модели для генерации ответа.
"""
from rag_example.simple_example.generators import get_base_grok_client, call_llm
from rag_example.simple_example.metrics import calculate_cosine_similarity
from rag_example.simple_example.retrievers import CORPUS
from rag_example.simple_example.utils import BaseTextPreprocessor, find_best_match_by_keywords

if __name__ == '__main__':
    prompt = "Режим работы Вкусно и точка на Московском"
    best_score, best_record = find_best_match_by_keywords(prompt, CORPUS)
    if best_record:
        similarity_score = calculate_cosine_similarity(prompt.split(), best_record.split(), BaseTextPreprocessor())
    client = get_base_grok_client()
    response = call_llm(client, "llama-3.3-70b-versatile", prompt, best_record)
    print(response.output_text)
