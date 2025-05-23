import pandas as pd
from pandas import DataFrame
from typing import Sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def get_dummy_variables_by_pandas(
        df: DataFrame,
        column_name: str,
        drop_first: bool = False
) -> DataFrame:
    """
    Метод для кодирования категориальных номинальных признаков в dataframe. Создает dummy-переменные.
    Args:
        df: dataframe с данными
        column_name: название переменной для кодирования
        drop_first: для получения k-1 dummy-переменных путем удаления первого столбца. True - да, False - нет
    Returns:
        Dataframe: dataframe с dummy-переменными
    """
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=drop_first)
    return dummies


def get_dummy_variables_by_scikit_learn(
        values: Sequence[Sequence[str]],
        nominal_feature_idx: int,
        other_features_idx: Sequence[int],
        title_features: Sequence[str] = None,
        drop_first: bool = False
) -> tuple[list[str], list[list[float]]]:
    """
    Метод для кодирования категориальных номинальных переменных с использованием scikit-learn.
    Args:
        values: список списков с признаками. Например, [['ball', 'red'], ['toy', 'black'], ['pen', 'white']]
        nominal_feature_idx: индекс номинального признака в подсписке values
        other_features_idx: индексы переменных в подсписке values, которые не нужно кодировать
        title_features: названия всех признаков в values. Длина title_features должна быть равной длине подсписка values
        drop_first: для получения k-1 dummy-переменных путем удаления первого столбца. True - да, False - нет
    Returns:
        Dataframe: dataframe с dummy-переменными
    """
    drop_first = 'first' if drop_first else None
    encoder = OneHotEncoder(drop=drop_first)
    transformer = ColumnTransformer([
        ('onehot', encoder, [nominal_feature_idx]),  # dummy-переменные по индексу nominal_feature_idx
        ('nothing', 'passthrough', other_features_idx)  # ничего не делаем с переменными по индексам other_features_idx
    ])
    dummies = transformer.fit_transform(values)
    field_names = transformer.get_feature_names_out(input_features=title_features)
    return field_names, dummies
