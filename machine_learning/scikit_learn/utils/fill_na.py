from pandas import DataFrame


def fill_na_quan_feature_by_mean(df: DataFrame, column_name: str) -> DataFrame:
    """
    Метод заполняет пустые значения в столбце column_name средним значением по данной колонке.
    Актуально, если выборка имеет нормальное распределение.
    Args:
        df: dataframe с данными
        column_name: название столбца (количественный признак)
    Returns:
        Dataframe: dataframe с заполненными значениями
    """
    df = df.copy(deep=True)
    df[column_name] = df[column_name].fillna(df[column_name].mean())
    return df


def fill_na_quan_feature_by_median(df: DataFrame, column_name: str) -> DataFrame:
    """
    Метод заполняет пустые значения в столбце column_name медианным значением по данной колонке.
    Актуально, если выборка имеет ненормальное распределение.
    Args:
        df: dataframe с данными
        column_name: название столбца (количественный признак)
    Returns:
        Dataframe: dataframe с заполненными значениями
    """
    df = df.copy(deep=True)
    df[column_name] = df[column_name].fillna(df[column_name].median())
    return df


def fill_na_qual_feature(df: DataFrame, column_name: str) -> DataFrame:
    """
    Метод заполняет пустые значения в столбце column_name значением, которое чаще всего встречаемся в этом столбце.

    Args:
        df: dataframe с данными
        column_name: название столбца (качественный признак)
    Returns:
        Dataframe: dataframe с заполненными значениями
    """
    df = df.copy(deep=True)
    df[column_name] = df[column_name].fillna(df[column_name].mode()[0])
    return df
