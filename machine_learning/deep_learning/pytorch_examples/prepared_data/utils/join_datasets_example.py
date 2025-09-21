import torch
from torch.utils.data import TensorDataset, Dataset


class CustomJointDatasets(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), 'The lengths of X and Y must be equal.'
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.x[idx], self.y[idx]


X = torch.normal(mean=0, std=1, size=(5, 4))
Y = torch.arange(5)
custom_joint_dataset = CustomJointDatasets(X, Y)
for i in custom_joint_dataset:
    print(i)

torch_joint_dataset = TensorDataset(X, Y)
for i in torch_joint_dataset:
    print(i)
