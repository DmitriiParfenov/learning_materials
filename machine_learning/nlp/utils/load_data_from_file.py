import pandas as pd
import os
from itertools import product


def get_df_from_file(basedir: str, permute: bool = False, max_rows: int = 2000) -> pd.DataFrame:
    """
    Загружает отзывы из директорий train/test и pos/neg по указанному пути basedir в dataframe.

    Параметры:
        base_dir (str): путь к корневой директории с поддиректориями train/test
        permute (bool): перемешивать ли строки внутри dataframe или нет
        max_rows (int): количество отзывов для выгрузки
    Возвращает:
        pd.DataFrame: dataframe с колонками ['review', 'sentiment'], где review - это отзыв пользователя, а sentiment -
            это класс
    """
    base_df = pd.DataFrame(columns=['review', 'sentiment'])
    if not os.path.exists(basedir):
        return base_df

    sample_type = ['test', 'train']
    review_type = {'pos': 1, 'neg': 0}
    raw_data = []

    for sample, review in product(sample_type, review_type):
        path_to_files = os.path.join(basedir, sample, review)
        for file in sorted(os.listdir(path_to_files))[:max_rows]:
            with open(os.path.join(path_to_files, file), encoding='UTF-8', mode='r') as read_file:
                raw_data.append({
                    'review': read_file.read(),
                    'sentiment': review_type[review]
                    })
    if not raw_data:
        return base_df
    if permute:
        # frac=1 - возвращаем все записи из df и изменяем индексы.
        return pd.DataFrame(raw_data).sample(frac=1).reset_index(drop=True)
    return pd.DataFrame(raw_data)