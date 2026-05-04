import re


class BaseTextPreprocessor:
    @staticmethod
    def remove_special_chars(text: list[str]) -> list[str]:
        pattern = r'[\W]+'
        return [re.sub(pattern, ' ', word.lower()) for word in text]

    @staticmethod
    def to_lowercase(text: list[str]) -> list[str]:
        return [word.lower().strip() for word in text]

    def normalize_text(self, text: list[str]) -> list[str]:
        return self.to_lowercase(self.remove_special_chars(text))
