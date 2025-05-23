from sklearn.neighbors import KNeighborsClassifier
from machine_learning.scikit_learn.prepared_data import prepared_data
import numpy as np
from machine_learning.scikit_learn.visualisation import plot_decision_regions
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()
    knn_model = KNeighborsClassifier(
        n_neighbors=10,  # k экземпляров
        weights="uniform",  # влияние ближайших соседей
        algorithm="auto",  # алгоритм поиска knn
        metric="euclidean",  # метрика расчета расстояния между точками
        n_jobs=2,  # количество процессоров
    )
    knn_model.fit(x_train, y_train)
    x_comb = np.vstack((x_train, x_test))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(x_comb, y_comb, knn_model, range(105, 150))
    plt.xlabel('Длина лепестка [см] ')
    plt.ylabel('Ширина лепестка [см] ')
    plt.legend(loc='upper left')
    plt.show()
