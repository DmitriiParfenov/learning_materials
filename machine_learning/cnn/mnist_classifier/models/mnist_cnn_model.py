import torch
import torch.nn as nn


class MnistCnnClassifier(nn.Module):
    """
    Сверточная нейронная сеть для классификации рукописных цифр. Данная сеть исключительно в целях демонстрации.
    Архитектура:
        1) Input layer: тензор формата [batch_size, channel, height, width].
        2) Convolutional layer: тензор формата [batch_size, height, width, count_feature_maps].
        3) Pooling layer: тензор формата [batch_size, height / P, width / P, count_feature_maps].
        4) Convolutional layer: тензор формата [batch_size, height, width, count_feature_maps].
        5) Pooling layer: тензор формата [batch_size, height / P, width / P, count_feature_maps].
        6) Flatten layer: вектор размера [batch_size * height / P * width / P * count_feature_maps].
        7) Full connected layer:
            7.1) Input: тензор [len(flatten_layer), n_hidden]
            7.2) Hidden layer: [n_hidden, n_classes]

    Параметры:
        1) Input layer предназначен для тензоров с 1 каналом.
        2) Convolutional layer выдает 32 карты активации. Размер фильтра 5*5, stride=1, padding=2, dilation=1.
        3) Pooling layer имеет ядро размером 2*2 и stride, равные размеру ядра.
        4) Convolutional layer выдает 64 карты активации. Размер фильтра 5*5, stride=1, padding=2, dilation=1.
        5) Pooling layer имеет ядро размером 2*2 и stride, равные размеру ядра.
        6) Flatten layer образует вектор с размером 7*7*64=3136.
        7) FC_1 - это скрытый слой с 1024 нейронами.
        8) FC_2 - это выходной слой с 10 нейронами.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1,  # количество каналов в input
            out_channels=32,  # количество карт активация = фильтров
            kernel_size=5,  # размер фильтра
            padding=2  # padding = same
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2)  # если не задавать stride, то он по умолчанию равен размеру ядра
        self.conv_2 = nn.Conv2d(
            in_channels=32,  # количество каналов из pooling layer
            out_channels=64,  # количество карт активация = фильтров
            kernel_size=5,  # размер фильтра
            padding=2  # padding = same
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(3136, 1024)  # 7*7*64=3136
        self.dropout = nn.Dropout(p=0.5)  # Регуляризация через dropout.
        self.fc_2 = nn.Linear(1024, 10)
        layers = [
            self.conv_1,
            nn.ReLU(),
            self.pool_1,
            self.conv_2,
            nn.ReLU(),
            self.pool_2,
            self.flatten,
            self.fc_1,
            nn.ReLU(),
            self.dropout,
            self.fc_2
        ]
        self.module_list = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.module_list:
            x = layer(x)
        return x
