import numpy as np
from sklearn.datasets import fetch_openml


def get_prepared_mnist_data() -> tuple[np.array, np.array]:
    x, y = fetch_openml(name='mnist_784', return_X_y=True, version=1)
    x = x.to_numpy()
    y = y.to_numpy(dtype=int)
    # Значения x переводим в диапазон от -1 до 1. Делаем это через центрирование вокруг нуля, а не через стандартизацию,
    # так как в нейросетях используют нормализацию (просто сжимаем выборку до нужного диапазона с сохранением пропорций)
    x = (x / x.max() - 0.5) * 2
    return x, y
