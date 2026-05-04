from rag_example.simple_example.generators import get_base_grok_client, call_llm
from rag_example.simple_example.retrievers import CORPUS
from rag_example.simple_example.utils import setup_vectorizer, find_best_match_by_tf_idf


def main(prompt: str) -> str:
    best_matched_record = ""
    vectorizer, tfidf_matrix = setup_vectorizer(CORPUS)
    best_score, best_record_index = find_best_match_by_tf_idf(prompt, vectorizer, tfidf_matrix)
    if best_score:
        best_matched_record = CORPUS[best_record_index]
    client = get_base_grok_client()
    response = call_llm(client, "llama-3.3-70b-versatile", prompt, best_matched_record)
    return response.output_text


if __name__ == '__main__':
    print(main("А ты сможешь сказать режим работы фаст фуда в СПБ?"))
