from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'wine_data.csv'


def get_wine_data(
        path_to_file: str = PATH
) -> tuple[DataFrame, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Метод возвращает данные из тестового набора wine:
    - df: dataframe с данными
    - x_train: обучающая выборка
    - x_train_std: стандартизированная обучающая выборка
    - x_test: тестовая выборка
    - x_test_std: стандартизированная тестовая выборка
    - y_train: метки для обучающей выборки
    - y_test: метки для тестовой выборки
    """
    df = pd.read_csv(path_to_file)
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.3,
        random_state=1,
        stratify=y
    )
    std_scaler = StandardScaler()
    x_train_std = std_scaler.fit_transform(x_train)
    x_test_std = std_scaler.transform(x_test)

    return df, x_train, x_train_std, x_test, x_test_std, y_train, y_test
