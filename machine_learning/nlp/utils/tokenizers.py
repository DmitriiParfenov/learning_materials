from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def simple_tokenizer(text: str) -> list[str]:
    """Просто разбивает текст по пробелам."""

    return text.split()


def porter_tokenizer(text: str) -> list[str]:
    """Разбиваем текст по пробелам и оставляем только основу слова из текста."""
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def porter_tokenizer_without_stop_words(text: str) -> list[str]:
    """Разбиваем текст по пробелам и оставляем только основу слова из текста. Также удаляем все стоп-слова."""

    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    words = [porter.stem(word) for word in text.split()]
    return [word for word in words if words not in stop_words]
