import time
from threading import Lock


class Throttle:
    """Класс для реализации алгоритма Token Bucket - в «ведро» с постоянной скоростью добавляются токены, а при
    обработке запроса токен из «ведра» удаляется; если же токенов не достаточно, то запрос отбрасывается."""

    def __init__(self, rate):
        """Экземпляр класса инициализируется:
        а) _consume_lock: чтобы несколько потоков не могли одновременно изменять данные о доступных токенах;
        б) tokens: хранение доступных токенов;
        в) rate: скорость выдачи токенов;
        г) last: момент времени последнего обновления количества токенов.
        """
        self._consume_lock = Lock()
        self.rate = rate
        self.tokens = 0
        self.last = 0

    def consume(self, amount=1):
        """Метод используется для получения токенов. В качестве аргумента он принимает количество запрашиваемых токенов
        (по умолчанию - 1)."""

        # Для безопасного использования метода блокировки.
        with self._consume_lock:
            now = time.time()

            # Сначала инициализируется измерение времени, чтобы избежать внутренних проблем.
            if self.last == 0:
                self.last = now

            # Вычисляем, сколько времени прошло с последнего обновления количества токенов.
            elapsed = now - self.last

            # Если прошло достаточно времени для добавления новых токенов, мы вычисляем количество токенов,
            # которые нужно добавить в корзину.
            if int(elapsed * self.rate):
                # Мы обновляем количество доступных токенов и время последнего обновления.
                self.tokens += int(elapsed * self.rate)
                self.last = now

            # Проверяем, не превышает ли текущее количество токенов заданную скорость. Если да, мы устанавливаем
            # количество токенов равным скорости.
            self.tokens = self.rate if self.tokens > self.rate else self.tokens

            # Если маркеры в наличии, то они выдаются.
            if self.tokens >= amount:
                self.tokens -= amount
            else:
                amount = 0

            return amount

