from datetime import datetime, timedelta

import numpy as np
from airflow import DAG
from airflow.sdk import task, TaskGroup
from numpy import floating


@task()
def generate_data() -> list[int]:
    return np.random.randint(1, 100, size=20).tolist()


@task()
def calc_avg(arr: list[int]) -> floating:
    return np.average(arr).round(2)


@task()
def calc_median(arr: list[int]) -> floating:
    return np.median(arr).round(2)


@task()
def calc_mode(arr: list[int]) -> float | None:
    values, counts = np.unique(arr, return_counts=True)
    if not values.size:
        return None
    modes = values[counts == np.max(counts)]
    return float(modes[0].round(2))


@task()
def show_results(avg, median, mode):
    print(f"AVG: {avg}")
    print(f"MEDIAN: {median}")
    print(f"MODE: {mode}")


default_args = {
    "owner": "Dmitrii Parfenov",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
        dag_id="calc_stats_with_task_group",
        default_args=default_args,
        start_date=datetime(2026, 4, 6),
        schedule="@daily",
        catchup=False,
):
    data = generate_data()
    with TaskGroup(group_id="statistics") as statistics:
        avg = calc_avg(data)
        median = calc_median(data)
        mode = calc_mode(data)
    show_results(avg, median, mode)
