import numpy as np

from machine_learning.deep_learning.utils import sigmoid, get_dummies


class NeuralNetMLP:
    """Класс для реализации многослойного перцептрона. Для примера взяли нейросеть с 1 скрытом слоем."""

    def __init__(self, n_features: int, n_hidden: int, n_classes: int, random_seed: int = 123) -> None:
        self.n_features = n_features  # кол-во признаков.
        self.n_hidden = n_hidden  # кол-во нейронов в скрытых слоях.
        self.n_classes = n_classes  # кол-во узлов в выходном слое.
        self.random_seed = random_seed

        rgen = np.random.RandomState(random_seed)
        # Веса и смещение для скрытого узла.
        self.weights_hidden = rgen.normal(loc=0.0, scale=0.01, size=(n_hidden, n_features))
        self.bias_hidden = np.zeros(n_hidden)

        # Веса и смещение для выходного узла
        self.weights_out = rgen.normal(loc=0.0, scale=0.01, size=(n_classes, n_hidden))
        self.bias_out = np.zeros(n_classes)

    def forward(self, x):
        z_hidden = x @ self.weights_hidden.T + self.bias_hidden
        a_hidden = sigmoid(z_hidden)

        z_out = a_hidden @ self.weights_out.T + self.bias_out
        a_out = sigmoid(z_out)
        return a_hidden, a_out

    def backward(self, x, a_hidden, a_out, y):
        # Кодируем y = [1, 2, 3] в y_dummies = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] при self.n_classes=4
        y_dummies = get_dummies(y, self.n_classes)

        # Обновляем веса и смещение для выходного слоя.
        # dL/dw_out = dL/da_out * da_out/dz_out * dz_out/dw_out
        # dL/db_out = dL/da_out * da_out/dz_out * dz_out/db_out
        dL__da_out = 2 * (a_out - y_dummies) / y.shape[0]
        da_out__dz_out = a_out * (1 - a_out)
        dz_out__dw_out = a_hidden
        delta_out = dL__da_out * da_out__dz_out
        # Обновленные веса и смещение выходного слоя.
        dL__dw_out = delta_out.T @ dz_out__dw_out
        dL__db_out = np.sum(delta_out, axis=0)

        # Обновляем веса и смещение для скрытого слоя.
        # dL/dw_h = dL/da_h * da_h/dz_h * dz_h/dw_h
        # dL/db_h = dL/da_h * da_h/dz_h * dz_h/db_h
        dL__da_h = delta_out @ self.weights_out  # dL/da_h = dL/da_out * da_out/dz_out * dz_out/da_h
        da_h__dz_h = a_hidden * (1 - a_hidden)
        dz_h__dw_h = x
        # Обновленные веса и смещение скрытого слоя.
        dL__dw_h = (dL__da_h * da_h__dz_h).T @ dz_h__dw_h
        dL__db_h = np.sum(dL__da_h * da_h__dz_h, axis=0)

        return dL__dw_out, dL__db_out, dL__dw_h, dL__db_h
