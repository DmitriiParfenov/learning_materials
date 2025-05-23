from sklearn.tree import DecisionTreeClassifier
from machine_learning.scikit_learn.prepared_data import prepared_data
from machine_learning.scikit_learn.visualisation import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Получили данные для модели.
    x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()

    # Обучение модели.
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree_model.fit(x_train, y_train)

    # Визуализация областей классов.
    x_comb = np.vstack((x_train, x_test))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(x_comb, y_comb, classifier=tree_model, test_idx=range(105, 150))
    plt.xlabel('Длина лепестка [см] ')
    plt.ylabel('Ширина лепестка [см] ')
    plt.legend(loc='upper left')
    plt.show()
