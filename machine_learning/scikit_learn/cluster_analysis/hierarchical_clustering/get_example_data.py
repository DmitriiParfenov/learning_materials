import numpy as np
import pandas as pd


def get_example_data(rows: int = 5, columns: int = 3) -> pd.DataFrame:
    """
    Метод возвращает dataframe с рандомными значениями.
    Args:
        rows: количество строк
        columns: количество колонок
    Returns:
        dataframe: dataframe с рандомными значениями
    """
    columns_names = [f'X_{x}' for x in range(1, columns + 1)]
    rows_names = [f'ID_{x}' for x in range(rows)]
    x = np.random.random_sample([rows, columns]) * 10
    return pd.DataFrame(data=x, index=rows_names, columns=columns_names)
