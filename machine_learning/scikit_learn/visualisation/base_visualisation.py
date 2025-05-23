import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    """
    Функция для визуализации границ решений классификатора для 2х признаков.
    Args:
        x: np.array - массив с 2мя признаками, например, np.array([[2, 3], [1, 1], [4, 5], [3, 2]])
        y: np.array - массив с откликами образцов, например, np.array([1, 0, 1, 0])
        classifier: объект модели для классификации данных
        test_idx: массив с индексами, чтобы выделить тестовые объекты из массива x
        resolution: шаг на графике.
    """
    # Устанавливаем цвета дял классов и образцов на графике.
    markers = ('o', 's', '^', 'v', '<')  # определяем вид образцов на графике
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # определяем цвет для каждого класса (максимально 5 классов)
    cmap = ListedColormap(colors[:len(np.unique(y))])  # задаем фиксированные цвета для каждого класса

    # Определяем длину оси x (первый признак) и длину оси y (второй признак).
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1  # min и max для первого признака (с небольшим отступом)
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # min и max для второго признака (с небольшим отступом)
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )  # построили сетку с min и max длинами с шагом = resolution.

    # Для каждой точки с шагом resolution в сетке определяем принадлежность к классу, чтобы закрасить области классов.
    # Данные для сетки преобразовали в одномерный массив и объединили в один двумерный массив с последующим
    # транспонированием, чтобы соотнести точки между собой для предсказания.
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)  # результаты предсказания превратили в двумерный массив (аналогично с xx2)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Рисуем точки на графике из x и y. Примечание: y == cl -> создает новые массив из булек.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],  # выбираем значение первого признака, если его метка равна метке cl
                    y=x[y == cl, 1],  # выбираем значение второго признака, если его метка равна метке cl
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # Выделяем тестовые образцы на графике.
    if test_idx:
        # Получаем тестовые точки из выборки по индексам из test_idx.
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0],
                    x_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')
