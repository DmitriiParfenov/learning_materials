"""
Строим диаграмму рассеивания признаков 'Длина чашелистика' и 'Длина лепестка' у растений Setosa и Versicolor. Если
визуально мы сможем провести линию между полученными данными, то перцептрон будет работать на этих данных (условно).
"""

import matplotlib.pyplot as plt
import numpy as np

from machine_learning.basic_algorithms.perceptron import TEST_DATAFRAME

# Создаем dataframe из файла: X1, X2, X3, X4, Y.
df = TEST_DATAFRAME
# Получили значения столбца 5 у строчек от 0 до 100.
y = df.iloc[0:100, 4].values
# Сделали кодирование отклика (если y = Iris-setosa, то в результате будет 0, иначе - 1).
y = np.where(y == 'Iris-setosa', 0, 1)
# Получили значения столбцов 0 И 2 у строчек от 0 до 100.
x = df.iloc[0:100, [0, 2]].values
# Отображаем данные.
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Длина чашелистика [см]')
plt.ylabel('Длина лепестка [см] ')
plt.legend(loc='upper left')
plt.show()
