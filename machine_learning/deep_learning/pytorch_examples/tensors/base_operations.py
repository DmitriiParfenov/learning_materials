import torch

tensor = torch.ones(size=(5, 4))  # Матрица 5 * 4
t_1 = torch.normal(mean=0, std=1, size=(5, 4))
t_2 = torch.normal(mean=1, std=2, size=(5, 4))

# Разделение тензора.
tensors = torch.chunk(tensor, 2, 1)  # разделили по оси 1, то есть по столбцу и получили 2 матрицы 5 * 2
for i in tensors:
    print(i)

# Конкатенация тензоров.
t = torch.cat([t_1, t_2], axis=1)  # соединили по столбцам: 5*4 + 5*4 = 5*8

# Стекирование тензоров.
t = torch.stack([t_1, t_2], axis=0)
print(t.shape)

