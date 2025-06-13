import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from machine_learning.scikit_learn.prepared_data import get_wine_data
from machine_learning.scikit_learn.visualisation import plot_decision_regions, visualize_features_impacts_for_pca
from machine_learning.scikit_learn.visualisation import visualize_dispersions

if __name__ == '__main__':
    df, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()
    lg_model = LogisticRegression(solver='lbfgs', random_state=1)
    pca_extractor = PCA(n_components=2, random_state=1)

    # Получаем новое k-мерное пространство признаков с 2 главными компонентами.
    x_train_std_pca = pca_extractor.fit_transform(x_train_std)
    x_test_std_pca = pca_extractor.transform(x_test_std)

    # Обучаем модель на основе новых признаков.
    lg_model.fit(x_train_std_pca, y_train)

    user_input = input(
        """Визуализация:
1. Визуализация границ решений классификатора на основе логистической регрессии
2. Дисперсия
3. Влияние признаков на главные компоненты
"""
    )
    if user_input == '1':
        x_comb = np.vstack((x_train_std_pca, x_test_std_pca))
        y_comb = np.hstack((y_train, y_test))
        plot_decision_regions(x_comb, y_comb, lg_model, range(105, 150))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='lower right')
        plt.show()
    elif user_input == '2':
        pca = PCA(n_components=None, random_state=1)  # n_components=None - для возврата всех компонент
        pca.fit(x_train_std)
        visualize_dispersions(eig_vals=pca.explained_variance_)
    elif user_input == '3':
        user_input = input('Выберете номер главной компоненты [1-13]: ')
        if user_input.isdigit() and int(user_input) in list(range(1, 14)):
            pca = PCA(n_components=None, random_state=1)
            pca.fit(x_train_std)
            sl_eigen_vals = pca.explained_variance_
            sl_eigen_vectors = pca.components_.T
            visualize_features_impacts_for_pca(
                df=df,
                idx_features=list(range(1, len(df.columns))),
                eig_vals=sl_eigen_vals,
                eig_vectors=sl_eigen_vectors,
                pc_number=int(user_input) - 1
            )
