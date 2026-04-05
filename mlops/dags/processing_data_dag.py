from datetime import datetime, timedelta

import numpy as np
from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.sdk import task


@task()
def generate_data() -> list[int]:
    return np.random.randint(1, 100, size=20).tolist()


@task()
def calculate_statistics(arr: list[int]) -> dict[str, float]:
    arr = np.array(arr)
    return {
        "min": arr.min().round(2),
        "mean": arr.mean().round(2),
        "max": arr.max().round(2),
        "std": arr.std(ddof=1).round(2),
        "median": np.median(arr).round(2),
    }


@task()
def upload_to_s3() -> None:
    s3_hook = S3Hook(aws_conn_id="minio")
    connection = s3_hook.list_keys(bucket_name="vcf")
    print("Files:", connection)


default_args = {
    "owner": "Dmitrii Parfenov",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        dag_id="processing_data",
        default_args=default_args,
        start_date=datetime(2026, 4, 4),
        schedule="@daily",
        catchup=False,
):
    generated_data = generate_data()
    stats = calculate_statistics(generated_data)
    upload_to_s3()
    print(stats)
