import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot


class DataCarLoader:
    @staticmethod
    def load_data(name: str) -> pd.DataFrame:
        auto_mpg = fetch_openml(name=name, version=1, as_frame=True)
        frame = auto_mpg.frame
        df = frame.dropna()
        return df.reset_index(drop=True)

    @staticmethod
    def random_split(
            df: pd.DataFrame,
            test_size: float = 0.2,
            shuffle: bool = True,
            random_state: int = 1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)

    @staticmethod
    def standardize_numeric_data(
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            numeric_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
        train_stats = df_train.describe().transpose()
        for column in numeric_columns:
            mean = train_stats.loc[column, 'mean']
            std = train_stats.loc[column, 'std']
            df_train_norm[column] = (df_train_norm[column] - mean) / std
            df_test_norm[column] = (df_test_norm[column] - mean) / std
        return df_train_norm, df_test_norm

    @staticmethod
    def one_hot_encoding(df: pd.DataFrame, columns: list[str], categories_map: dict[str, int]) -> torch.Tensor:
        tensors = []
        for column in columns:
            unique_values = df[column].unique()
            encoded_values = {value: index for index, value in enumerate(unique_values)}
            new_column = f'{column}_encoded'
            df[new_column] = df[column].map(encoded_values)
            tensors.append(one_hot(torch.from_numpy(df[new_column].to_numpy(int)), num_classes=categories_map[column]))
        return torch.concat(tensors, 1)

    @staticmethod
    def get_ranked_values(df: pd.DataFrame, column_name: str, bins: list[int]) -> torch.Tensor:
        """
        Метод группирует значения по группам (аналогично pd.cut()). Используется torch.bucketsize - метод разделяет
        значения по группам и возвращает индексы, к которым принадлежат значения из values.
        Пример смотри в deep_learning/pytorch_examples/tensors/rank.py.

        Args:
            df: датафрейм с данными;
            column_name: название колонки, данные в которой нужно группировать;
            bins: список групп.
        Returns:
            tensor: тензор с индексами bins, к которым принадлежат значения из колонки columns.
        """
        return torch.bucketize(
            input=torch.from_numpy(df[column_name].to_numpy(int)),
            boundaries=torch.tensor(bins),
            right=True  # правая граница включительно.
        )

    def get_prepared_data(
            self, name: str,
            numeric_columns: list[str],
            categorical_columns: list[str],
            test_size: float = 0.2,
            shuffle: bool = True,
            random_state: int = 1
    ):
        df = self.load_data(name)
        categories_map = {column: len(df[column].unique()) for column in categorical_columns}
        df_train, df_test = self.random_split(df, test_size, shuffle, random_state)
        df_train_norm, df_test_norm = self.standardize_numeric_data(df_train, df_test, numeric_columns)

        # Код для примера группировки колонки модели по годам - (-∞, 73), [73, 76), [76, 79) и [76, ∞).
        df_train_norm['model_decoded'] = self.get_ranked_values(df_train_norm, 'model', [73, 76, 79])
        df_test_norm['model_decoded'] = self.get_ranked_values(df_test_norm, 'model', [73, 76, 79])
        numeric_columns.append('model_decoded')

        x_train_num, x_test_num = (
            torch.from_numpy(df_train_norm[numeric_columns].to_numpy(dtype=float)),
            torch.from_numpy(df_test_norm[numeric_columns].to_numpy(dtype=float))
        )
        x_train_cat, x_test_cat = (
            self.one_hot_encoding(df_train, categorical_columns, categories_map),
            self.one_hot_encoding(df_test, categorical_columns, categories_map)
        )
        y_train, y_test = (
            torch.from_numpy(df_train['class'].to_numpy(float)).float(),
            torch.from_numpy(df_test['class'].to_numpy(float)).float(),
        )
        x_train, x_test = (
            torch.concat([x_train_num, x_train_cat], 1).float(),
            torch.concat([x_test_num, x_test_cat], 1).float()
        )
        return x_train, x_test, y_train, y_test
