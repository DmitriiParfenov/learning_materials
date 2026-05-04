from typing import Protocol


class TextPreprocessorProtocol(Protocol):
    @staticmethod
    def remove_special_chars(text: list[str]) -> list[str]: ...

    @staticmethod
    def to_lowercase(text: list[str]) -> list[str]: ...

    def normalize_text(self, text: list[str]) -> list[str]: ...
