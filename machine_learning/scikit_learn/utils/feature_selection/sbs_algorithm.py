from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from machine_learning.scikit_learn.prepared_data import get_wine_data
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



class SBS:
    """
    Класс реализует жадный алгоритм отбора признаков - Sequential Backward Selection.
    Цель - уменьшение размерности признакового пространства с минимальным снижением эффективности классификатора.
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            k_features: int,
            scoring=accuracy_score,
            random_state: int = 1,
            test_size: float = 0.25
    ) -> None:
        self.estimator = clone(estimator)  # Классификатор.
        self.k_features = k_features  # Финальное кол-во признаков.
        self.scoring = scoring  # Метод для оценки классификации
        self.random_state = random_state
        self.test_size = test_size

        self._features_subsets = []  # Храним все индексы признаков во время их отбора.
        self._features_scores = []  # Храним оценки предсказаний во время отбора признаков.
        self.subset = None  # Индексы признаков после отбора лучших признаков в размере k_features.
        self.score = None  # Оценка предсказания модели после отбора лучших признаков в размере k_features.

    def fit(self, x: np.array, y: np.array):
        """
        Метод сводит начальное d-мерное пространство признаков к k-мерному подпространству признаков, где k <d.
        Алгоритм:
            1) Инициализировали признаковое пространство.
            2) Разбили выборку на обучающую и тестовую. Оценили качество модели по всем признакам.
            3) Запускаем цикл до тех пор, пока количество признаков не достигнет k_features.
                - убираем по одному признаку и проверяем прогностическую способность модели (для каждого признакового
                подпространства мы смотрим все комбинации признаков)
                - выбираем то подпространство, которое дает самый лучший прогноз
        """
        # Получение обучающей и тестовой выборок.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            random_state=self.random_state,
            test_size=self.test_size,
            stratify=y
        )
        # Инициализация признаков и оценки модели.
        initial_feature_count = x_train.shape[1]  # Общее кол-во признаков.
        initial_feature_idx = tuple(range(initial_feature_count))  # Получаем индексы каждого признака.
        initial_feature_score = self.calculate_score(x_train, x_test, y_train, y_test, initial_feature_idx)

        # Сохраняем индексы признаков и оценку модели в атрибуты экземпляра класса.
        self._features_subsets.append(initial_feature_idx)  # Список всех индексов признаков.
        self._features_scores.append(initial_feature_score)  # Список оценок моделей при текущем кол-ве признаков.
        self.subset = initial_feature_idx  # Для доступа к финальному кол-ву признаков по их индексам.
        self.score = initial_feature_score  # Для доступа к финальной оценке модели с уменьшенным количеством признаков.

        while initial_feature_count > self.k_features:
            scores = []
            subsets = []

            # Оцениваем модель при уменьшении количества признаков на 1 с использованием combinations - метод, который
            # возвращает все комбинации признаков между собой в количестве r.
            for subset in combinations(initial_feature_idx, r=initial_feature_count - 1):
                current_score = self.calculate_score(x_train, x_test, y_train, y_test, subset)
                scores.append(current_score)
                subsets.append(subset)

            # Выбираем ту комбинацию признаков, которую дала самую лучшую оценку прогнозирования.
            best_score_idx = np.argmax(scores)
            best_score = scores[best_score_idx]
            idx_of_features = subsets[best_score_idx]

            self._features_subsets.append(idx_of_features)
            self._features_scores.append(best_score)
            self.subset = idx_of_features
            self.score = best_score

            initial_feature_count -= 1

        return self

    def transform(self, x):
        """Метод сводит начальное d-мерное пространство признаков к k-мерному подпространству признаков, где k <d."""

        return x[:, self.subset]

    def calculate_score(
            self,
            x_train: np.array,
            x_test: np.array,
            y_train: np.array,
            y_test: np.array,
            indices: Sequence[int]
    ) -> float | int:
        """
        Метод осуществляет оценку качества модели по переданным данным. Количество признаков определяется индексами
        признаков в indices.
        """
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_true=y_test, y_pred=y_pred)
        return score


if __name__ == '__main__':
    df_wine, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Уменьшаем размер признакового пространства до 2.
    sbs = SBS(estimator=knn_model, k_features=1)
    sbs.fit(x_train_std, y_train)
    features_subsets = sbs._features_subsets  # Получили индексы признаков при уменьшении размерности признаков (13-1)
    k4 = list(features_subsets[-4])  # По графику от 2 до 9 признаков дают 100 % точность. На рандом выбрали 4 признака
    title_k4 = df_wine.columns[1:][k4].values

    # Качество модели до уменьшения признакового пространства.
    knn_model.fit(x_train_std, y_train)
    score_train_before = round(knn_model.score(x_train_std, y_train), 4)
    score_test_before = round(knn_model.score(x_test_std, y_test), 4)

    # Качество модели после уменьшения признакового пространства.
    knn_model.fit(x_train_std[:, k4], y_train)
    score_train_after = round(knn_model.score(x_train_std[:, k4], y_train), 4)
    score_test_after = round(knn_model.score(x_test_std[:, k4], y_test), 4)

    features_count = [len(feature) for feature in sbs._features_subsets]
    scores = sbs._features_scores

    plt.plot(features_count, scores, marker='o')
    plt.xlim([0, 15])
    plt.ylim([0.5, 1.05])
    plt.grid()
    plt.title('Уменьшение признакового пространства')
    plt.ylabel('Точность, доли')
    plt.xlabel('Кол-во признаков, шт')
    plt.show()

