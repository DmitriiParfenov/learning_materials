from typing import Generator

import numpy as np


class NeuralNetHelper:
    @classmethod
    def get_dummies(cls, y: np.ndarray, labels: int = 10) -> np.ndarray:
        dummies = np.zeros(shape=(y.shape[0], labels))
        for idx, num in enumerate(y):
            dummies[idx, num] = 1
        return dummies

    @classmethod
    def get_minibatch_generator(
            cls,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 100,
            shuffle: bool = True
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        indexes = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(indexes)

        for idx in range(0, len(x) - 1, batch_size):
            yield x[idx:idx + batch_size], y[idx:idx + batch_size]

    @classmethod
    def calculate_accuracy(cls, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        return np.mean(y_true == y_pred)

    def calculate_mse(self, y_true: np.ndarray, probas: np.ndarray, labels: int = 10) -> np.floating:
        encoded_y_true = self.get_dummies(y_true, labels)
        return np.mean((encoded_y_true - probas) ** 2)

    def calc_mse_and_acc(
            self,
            neural_net,
            x: np.ndarray,
            y_true: np.ndarray,
            batch_size: int = 100,
            shuffle: bool = True,
            labels: int = 10
    ) -> tuple[float, float]:
        """
        Массивы x и y_true разделяем на мини-батчи. Каждый батч подаем нейросети neural_net и рассчитываем mse и
        accuracy. После того как через всю neural_net провели все мини-батчи, рассчитываем общие MSE и accuracy для
        всего набора x и y_true.
        """
        batch_generator = self.get_minibatch_generator(x, y_true, batch_size, shuffle)
        mse, corrected_pred, count_batches, n_samples = 0, 0, 0, 0
        for x_mini, y_mini in batch_generator:
            _, probas = neural_net.forward(x_mini)  # probas - массив с вероятностями принадлежности к опред. классам.
            # Расчет MSE для мини-батча. Тотальный MSE будет равен: mse / count_batches
            mse += self.calculate_mse(y_mini, probas, labels)
            count_batches += 1
            # Расчет accuracy. Тотальный accuracy будет равен: corrected_pred / n_samples
            # decoded_y_pred - здесь получаем число от 0 до 9, которое имеет самую максимальную вероятность.
            decoded_y_pred = np.argmax(probas, axis=1)
            # corrected_pred - количество верно предсказанных классов, а n_samples - общее кол-во примеров в мини-batch.
            corrected_pred += (y_mini == decoded_y_pred).sum()
            n_samples += x_mini.shape[0]

        return mse / count_batches, corrected_pred / n_samples

    def train(
            self,
            neural_net,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            learning_rate: float = 0.01,
            epochs: int = 50,
            batch_size: int = 100,
            shuffle: bool = True,
            labels: int = 10
    ):
        epoch_loss = []  # собираем потери при обучении.
        epoch_train = []  # точность после обучения на обучающей выборке.
        epoch_test = []  # точность после обучения на тестовой выборке.

        for e in range(epochs):
            batch_generator = self.get_minibatch_generator(x_train, y_train, batch_size, shuffle)
            for x_mini, y_mini in batch_generator:
                # Прямое распространение сигнала.
                a_h, a_out = neural_net.forward(x_mini)
                # Обратное распространение ошибки.
                dl_dw_out, dl_db_out, dl_dw_h, dl_db_h = neural_net.backward(x_mini, a_h, a_out, y_mini)
                # Обновление весов у нейросети.
                neural_net.weights_out -= learning_rate * dl_dw_out
                neural_net.bias_out -= learning_rate * dl_db_out
                neural_net.weights_hidden -= learning_rate * dl_dw_h
                neural_net.bias_hidden -= learning_rate * dl_db_h

            train_mse, train_acc = self.calc_mse_and_acc(neural_net, x_train, y_train, batch_size, shuffle, labels)
            test_mse, test_acc = self.calc_mse_and_acc(neural_net, x_test, y_test, batch_size, shuffle, labels)

            epoch_loss.append(train_mse)
            epoch_train.append(train_acc * 100)
            epoch_test.append(test_acc * 100)
            print(f'Epoch: {e + 1:03d}/{epochs:03d} '
                  f'| Train MSE: {train_mse:.2f} '
                  f'| Train Acc: {train_acc * 100:.2f}% '
                  f'| Test Acc: {test_acc * 100:.2f}%')

        return epoch_loss, epoch_train, epoch_test
