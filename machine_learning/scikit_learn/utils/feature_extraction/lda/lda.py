import numpy as np

from machine_learning.scikit_learn.prepared_data import get_wine_data
import matplotlib.pyplot as plt

from machine_learning.scikit_learn.visualisation import (
    visualize_discriminants,
    visualize_new_lda_frame
)


def main() -> None:
    df, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()

    d = x_train_std.shape[1]  # количество признаков.
    S_W = np.zeros((d, d))  # Нулевая матрица для внутриклассового разброса.
    S_B = np.zeros((d, d))  # Нулевая матрица для межклассового разброса.

    # Определяем внутриклассовый разброс.
    for label in np.unique(y_train):
        S_W += np.cov(x_train_std[y_train == label].T)

    # Определяем межклассовый разброс.
    mean_feature_by_class = []  # Средние значения признаков для каждого класса.
    for label in np.unique(y_train):
        mean_feature_by_class.append(np.mean(x_train_std[y_train == label], axis=0))

    mean_feature_overall = np.mean(x_train_std, axis=0)  # Средние значения признаков в рамках всего набора.
    mean_feature_overall = mean_feature_overall.reshape(d, 1)  # Чтобы делать матричное произведение.

    for label, mean_feature in enumerate(mean_feature_by_class, start=1):
        n_k = x_train_std[y_train == label].shape[0]  # Количество объектов в классе.
        mean_feature = mean_feature.reshape(d, 1)  # Из [1, 2, 3] в [[1, 2, 3]]
        S_B += n_k * (mean_feature - mean_feature_overall).dot((mean_feature - mean_feature_overall).T)

    # Разложение на собственные значения и собственные вектора.
    eigen_vals, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_vals = eigen_vals.real
    eigen_vectors = eigen_vectors.real
    eigen_pairs = [(abs(eigen_vals[idx]), eigen_vectors[:, idx]) for idx in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # Получаем вектор проекций w.
    count_discriminants = len(np.unique(y_train)) - 1
    w = np.hstack([pairs[1][:, np.newaxis] for pairs in eigen_pairs[:count_discriminants]])  # Матрица проекций W.
    x_train_std_lda = x_train_std.dot(w)  # Новое k-мерное пространство признаков.

    user_input = input(
        """Визуализация:
1. Линейные дискриминанты
2. Разделение классов в новом подпространстве
"""
    )
    if user_input == '1':
        visualize_discriminants(eigen_vals)
    elif user_input == '2':
        visualize_new_lda_frame(x_train_std_lda, y_train)


if __name__ == '__main__':
    main()
