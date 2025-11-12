import torch.nn as nn
from functorch.dim import Tensor


class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, rnn_hidden_size: int, fc_hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden_size,  # кол-во нейронов в рекуррентном слое - размер вектора ht.
            num_layers=2,  # 2 рекуррентных слоя.
            batch_first=True
        )
        self.fc_1 = nn.Linear(in_features=rnn_hidden_size, out_features=fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text: str, lengths: Tensor):
        # Формируем тензор с входным признаками.
        x = self.embedding(text)
        # Последовательности выравнивали нулями до одинаковой длины. При обучении необходимо передать настоящие длины
        # последовательностей без padding, чтобы работать с реальными данными. lengths необходимо перевести на CPU -
        # это ограничение pytorch
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (ht, ct) = self.rnn(x)
        ht = ht[-1, :, :]  # так как у нас 2 rnn слоя, то берем только выходных значения
        x = self.fc_1(ht)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return x
