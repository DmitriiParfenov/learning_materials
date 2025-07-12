import re

from bs4 import BeautifulSoup


class TextCleaner:
    """Класс пример для очистки текстовых данных."""

    @classmethod
    def delete_html(cls, text: str) -> str:
        """Метод для удаления HTML-разметки из текста text."""

        return BeautifulSoup(text, "html.parser").get_text()

    @classmethod
    def remove_special_chars(cls, text: str, pattern: str = r'[\W]+') -> str:
        """
        Метод для удаления специальных символов из текста text через regex.
        Паттерн по умолчанию: '[\W]+', который находит последовательности букв, цифр и нижних подчёркиваний (инверсия).
        """
        return re.sub(pattern, ' ', text.lower())

    def clean_text(self, text: str):
        text = self.delete_html(text)
        text = self.remove_special_chars(text)
        return text
