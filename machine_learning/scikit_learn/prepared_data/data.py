import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepared_data() -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Метод возвращает данные из тестового набора iris:
    - x_train: обучающая выборка
    - x_train_std: стандартизированная обучающая выборка
    - x_test: тестовая выборка
    - x_test_std: стандартизированная тестовая выборка
    - y_train: метки для обучающей выборки
    - y_test: метки для тестовой выборки
    """
    # Получаем тестовые данные iris из самой библиотеки.
    iris = load_iris()
    x = iris.data[:, [2, 3]]  # признаки.
    y = iris.target  # отклики (3 отклика - 0, 1 и 2).

    # Разделяем набор данных на тестовые и обучающие.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.3,  # 30 % - это тестовые данные.
        random_state=1,  # до разбиения наборы будут перемешаны (shuffle).
        stratify=y  # Если отклики в y = [0 1 2] были в соотношении 25% / 40% / 35%, то в обучающем и тестовом наборе
        # эти же пропорции сохранятся.
    )

    # Стандартизируем значения признаков - (xi - xmean) / std.
    sc = StandardScaler()
    sc.fit(x_train)  # Установили выборочное среднее значение и Std.
    x_train_std = sc.transform(x_train)  # Стандартизовали x_train.
    x_test_std = sc.transform(x_test)  # Стандартизовали x_test.

    return x_train, x_train_std, x_test, x_test_std, y_train, y_test