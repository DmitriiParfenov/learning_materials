import torch
import numpy as np

data = [[1, 2], [6, 5], [7, 8]]

tensor_1 = torch.tensor(data, dtype=torch.float32)
tensor_2 = torch.tensor(data, dtype=torch.float32)


t_1 = torch.multiply(tensor_1, tensor_2)  # поэлементное умножение
t_2 = t_1.mean()  # среднее значение всех элементов тензора
t_3 = tensor_1 @ tensor_2.T  # матричное произведение
t_4 = torch.matmul(tensor_1, torch.transpose(tensor_2, 0, 1))  # матричное произведение (аналогично)
t_5 = torch.linalg.norm(tensor_1, ord=2, dim=1)  # L2-норма - SQRT(x^2 + y^2)
t_6 = torch.linalg.norm(tensor_1, ord=1, dim=1)  # L1-норма - x^2 + y^2
