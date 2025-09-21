import numpy as np
import torch

array_by_numpy = np.array([1, 2, 3], dtype=np.int32)
array_by_python = [4, 5, 6]

# Создание тензоров.
tensor_a = torch.from_numpy(array_by_numpy)  # создали тензор из numpy.
tensor_b = torch.tensor(array_by_python)  # создали тензор из списка.
tensor_c = torch.rand(3, 5)  # создали рандомный двумерный тензор 3 * 5
tensor_d = torch.ones(3, 4, 5)  # создали трехмерный тензор, состоящий из единиц
tensor_f = torch.tensor([
    [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]],
    [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5]],
    [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5]]
])  # трехмерный тензор, у которого 3 оси (ось 0 = 3 матрицы, ось 1 = 5 строк, ось 2 = 2 столбца)

torch.manual_seed(1)  # чтобы тензоры с рандомными числами всегда были одинаковыми
tensor_g = 2 * torch.rand(3, 5, 2) - 1  # тензор со значениями в диапазоне от 0 до 1
tensor_h = torch.normal(mean=0, std=1, size=(5, 2, 3))  # тензор со стандартным нормальным распределением