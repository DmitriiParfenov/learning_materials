import numpy as np

from machine_learning.scikit_learn.prepared_data.wine_data import get_wine_data
from machine_learning.scikit_learn.visualisation import (
    visualize_new_pca_frame,
    visualize_dispersions,
    visualize_features_impacts_for_pca,
)

if __name__ == '__main__':
    df, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()
    cov_matrix = np.cov(x_train_std.T)  # Транспонирование нужно, так как строки - это образцы, а столбцы - признаки
    eigen_vals, eigen_vectors = np.linalg.eig(cov_matrix)  # Собственные вектора (PC) и значения (дисперсии)
    eigen_pairs = [(abs(eigen_vals[idx]), eigen_vectors[:, idx]) for idx in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))  # Матрица проекций W.
    x_train_pca = x_train_std.dot(w)  # Новое k-мерное пространство признаков.

    user_input = input(
        """Визуализация:
1. Дисперсия
2. Новое k-мерное пространство признаков
3. Влияние признаков на главные компоненты
"""
    )
    if user_input == '1':
        visualize_dispersions(eigen_vals)
    elif user_input == '2':
        visualize_new_pca_frame(x_train_pca, y_train)
    elif user_input == '3':
        user_input = input('Выберете номер главной компоненты [1, 2]: ')
        if user_input.isdigit() and user_input in ('1', '2'):
            visualize_features_impacts_for_pca(
                df=df,
                idx_features=list(range(1, len(df.columns))),
                eig_vals=eigen_vals,
                eig_vectors=eigen_vectors,
                pc_number=int(user_input) - 1
            )
