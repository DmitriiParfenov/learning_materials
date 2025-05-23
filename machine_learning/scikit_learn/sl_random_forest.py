from sklearn.ensemble import RandomForestClassifier
from machine_learning.scikit_learn.prepared_data import prepared_data
import numpy as np
from machine_learning.scikit_learn.visualisation import plot_decision_regions
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()
    forest_model = RandomForestClassifier(
        n_estimators=25,  # Количество деревьев.
        n_jobs=2,  # Количество процессоров для работы.
        random_state=1,
    )
    forest_model.fit(x_train, y_train)
    x_comb = np.vstack((x_train, x_test))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(x_comb, y_comb, forest_model, range(105, 150))
    plt.xlabel('Длина лепестка [см] ')
    plt.ylabel('Ширина лепестка [см] ')
    plt.legend(loc='upper left')
    plt.show()

