from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig

from rag_example.core.s3_module.s3_error_handler import s3_error_handler


class S3Client:
    def __init__(self, access_key: str, secret_key: str, endpoint_url: str, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @s3_error_handler(default=False)
    def file_exists(self, file_key: str) -> bool:
        self.s3_client.head_object(Bucket=self.bucket_name, Key=file_key)
        return True

    @s3_error_handler(default=False)
    def download_file(self, file_key: str, filename: str) -> bool:
        self.s3_client.download_file(self.bucket_name, file_key, filename)
        return True

    @s3_error_handler(default=False)
    def upload_file(
            self,
            filename: str,
            file_key: str | None = None,
            chunk_size: int = 8,
            extra_args: dict[str, Any] | None = None
    ) -> bool:
        filename = Path(filename)
        if not filename.exists() and not filename.is_file():
            return False
        file_key = file_key if file_key else filename.name
        config = TransferConfig(multipart_chunksize=chunk_size * 1024 * 1024)
        self.s3_client.upload_file(
            Bucket=self.bucket_name,
            Key=file_key,
            Filename=filename,
            Config=config,
            ExtraArgs=extra_args
        )
        return True
