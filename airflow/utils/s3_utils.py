from pathlib import Path

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def upload_to_s3(filename: Path | str, bucket_name: str, key_name: str) -> None:
    hook = S3Hook(aws_conn_id="minio")
    hook.load_file(
        filename=filename,
        key=key_name,
        bucket_name=bucket_name,
        replace=True
    )


def download_from_s3(bucket_name: str, key_name: str, local_path: str | None = None) -> None:
    hook = S3Hook(aws_conn_id="minio")
    hook.download_file(
        key=key_name,
        bucket_name=bucket_name,
        local_path=local_path
    )

