import pandas as pd


def prepared_ames_housing_data(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Метод возвращает dataframe только с указанными столбцами из columns, если их передали.
    Args:
        df: dataframe с данными
        columns: список с названиями колонок
    Returns:
        df: dataframe с данными

    """
    existed_columns = set(df.columns)
    if not existed_columns.issuperset(columns):
        raise ValueError(f'Columns {columns} not found')

    df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})

    if columns:
        return df[columns]
    return df
