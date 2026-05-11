from typing import Protocol, Any


class IS3Client(Protocol):
    def file_exists(self, filename: str) -> bool: pass
    def download_file(self, file_key: str, filename: str) -> bool: pass
    def upload_file(
            self,
            filename: str,
            file_key: str | None = None,
            chunk_size: int = 8,
            extra_args: dict[str, Any] | None = None
    ) -> bool: pass
