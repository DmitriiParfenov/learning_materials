import re
from typing import Protocol


class BaseTextCleaner:
    @staticmethod
    def remove_special_chars(content: str) -> str:
        content = re.sub(r"\[\d+]", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        return content

    @staticmethod
    def to_lowercase(content: str) -> str:
        return content.lower()

    def normalize_text(self, content: str) -> str:
        return self.to_lowercase(self.remove_special_chars(content))
