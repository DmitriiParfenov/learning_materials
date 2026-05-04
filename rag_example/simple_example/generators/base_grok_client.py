from openai import OpenAI
from openai.types.responses import Response

from rag_example.simple_example.config import settings
from rag_example.simple_example.utils import timer


def get_base_grok_client(*args, **kwargs) -> OpenAI:
    return OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROK_BASE_URL, *args, **kwargs)


@timer
def call_llm(client: OpenAI, model: str, prompt: str, context: str = "") -> Response:
    prompt = f"Контекст: {context}, Вопрос пользователя {prompt}"
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Answer ONLY in Russian. Do not mix languages."},  # LLM пишет на русском.
            {"role": "system", "content": "Ты эксперт в NLP."},  # Теперь LLM считается экспертом в NLP
            {
                "role": "system",
                "content": "Ты ассистент, который отвечает строго на основе предоставленного контекста (КОНТЕКСТ)."
            },
            {"role": "user", "content": prompt}  # Реальный вопрос от пользователя.
        ],
        max_output_tokens=512
    )
    return response
