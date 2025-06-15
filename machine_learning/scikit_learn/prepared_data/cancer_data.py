from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_cancer_data():
    """
    Метод возвращает данные из тестового набора breast_cancer:
    - df: dataframe с данными
    - x_train: обучающая выборка
    - x_test: тестовая выборка
    - y_train: метки для обучающей выборки
    - y_test: метки для тестовой выборки
    """

    cancer_data = load_breast_cancer(as_frame=True)
    df = cancer_data.frame

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        stratify=y,
        random_state=1,
        test_size=0.2
    )

    return df, x_train, x_test, y_train, y_test
