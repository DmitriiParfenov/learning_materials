from torch.utils.data import DataLoader
import torch

tensor = torch.normal(mean=0, std=1, size=(4, 5, 3), dtype=torch.float)
data_loader = DataLoader(
    tensor,  # данные: мб списки, массивы numpy, тензоры
    batch_size=2,  # пакеты, размером в 2 тензора
    shuffle=True,
    drop_last=False
)
for t in data_loader:
    print(t)