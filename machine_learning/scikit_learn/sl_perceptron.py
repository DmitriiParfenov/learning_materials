import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from machine_learning.scikit_learn.prepared_data import prepared_data

x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()

# Обучаем модель Перцептрон
perceptron = Perceptron(eta0=0.1, random_state=1)  # Объявили объект модели.
perceptron.fit(x_train_std, y_train)  # Обучение.
y_pred = perceptron.predict(x_test_std)  # Получение предсказательных данных.

# Расчет точности модели вручную.
error_pred: np.ndarray = y_pred != y_test  # [False, True, False, ...] True -> y_pred отличается от y_true для образца.
count_error_pred = error_pred.sum()  # Складывает все bool в массиве (False = 0, True = 1).
accuracy_manual = round(1 - count_error_pred / y_test.size, 2)  # Точность модели: 1 - ошибка (неверные / все).

# Расчет точности модели.
accuracy_auto = accuracy_score(
    y_true=y_test,
    y_pred=y_pred
)  # Точность модели: 1 - ошибка (кол-во неверных предсказаний / все отклики).

print(perceptron.coef_)  # веса
print(perceptron.intercept_)  # смещение
