from typing import Protocol


class TextCleanerProtocol(Protocol):
    @staticmethod
    def remove_special_chars(content: str) -> str: pass

    @staticmethod
    def to_lowercase(content: str) -> str: pass

    def normalize_text(self, content: str) -> str: pass
