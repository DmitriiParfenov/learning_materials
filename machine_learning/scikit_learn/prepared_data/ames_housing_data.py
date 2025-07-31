import os

import pandas as pd


def get_ames_housing_data(pathway: str, delimiter: str = ',', encoding: str = 'UTF-8') -> pd.DataFrame:
    if not os.path.exists(pathway):
        raise FileNotFoundError('File not found')

    return pd.read_csv(filepath_or_buffer=os.path.join(pathway), delimiter=delimiter, encoding=encoding)
