from machine_learning.basic_algorithms.perceptron import (
    TEST_DATAFRAME,
    Perceptron
)
import numpy as np

# Получаем 2 (рандомных) признака из TEST_DATAFRAME.
x = TEST_DATAFRAME.iloc[0:100, [0, 2]].values

# Получаем отклики для двух рандомных признака из TEST_DATAFRAME.
y = TEST_DATAFRAME.iloc[0:100, 4].values
# Проводим кодирование отклика (если y = Iris-setosa, то в результате будет 0, иначе - 1).
y = np.where(y == 'Iris-setosa', 0, 1)

# Создаем instance Perceptron и обучаем модель на основе данных x и y.
perceptron = Perceptron(rate=0.01, epoch=100)
perceptron.fit(x, y)

# Теперь пробуем предсказать данные для набора данных.
data = [(5.1, 1.4), (6.4, 4.5), (13.1, 54.2)]  # Должно быть: 0 -> 1 -> пробуем узнать
for item in data:
    test_data = np.array(item)
    predicted_data = perceptron.predict(test_data)
    result = 'Это Iris-setosa' if predicted_data == 0 else 'Это Iris-versicolor'
    print(result)
