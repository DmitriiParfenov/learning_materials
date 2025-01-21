from threading import Thread
import socket


class ClientEchoThread(Thread):
    """
    Класс, который позволяет запускать многопоточный эхо сервер. Для каждого подключившегося клиента будет создан
    отдельный поток, и в этом потоке будет происходить чтение и запись данных.
    """

    def __init__(self, client: socket.socket) -> None:
        """Экземпляр класса инициализирует сокет клиента, через который сервер будет обмениваться данными."""
        super().__init__()
        self.client = client

    def run(self):
        """Переопределили метод run у класса Thread. Это метод сразу же запускается при запуске потока. Он в бесконечном
        цикле принимаем данные от клиента и в ответе отправляем Hello World!"""
        try:
            while True:
                # Получаем данные от клиента (может быть HTTP-запрос). В переменную data записывается до 2048 байт.
                data: bytes = self.client.recv(2048)
                # Если подключение было закрыто клиентом или остановлено сервером, то рейзим BrokenPipeError, чтобы
                # потом перехватить в except OSError для корректного закрытия сокета.
                if not data:
                    raise BrokenPipeError('Подключение закрыто!')
                print(f'Получено сообщение: {data.decode(encoding="utf-8")}')
                # Отправляем данные клиенту.
                self.client.sendall(bytes('Hello World!', encoding='utf-8'))
        except OSError as e:
            # Возбуждается методом sendall в случае закрытия клиентского сокета, при этом весь поток завершается.
            print(f'Поток прерван исключением {e}, производится остановка!')

    def close(self):
        """
        Метод позволяет корректно завершить активный поток с подключенным сокетом (активный поток - это поток, у
        которого работает метод run).
        """
        if self.is_alive():
            self.client.sendall(bytes('Останавливаюсь!', encoding='utf-8'))
            self.client.shutdown(socket.SHUT_RDWR)


if __name__ == '__main__':
    # Здесь создаем сокет server для работы с интернет-протоколом (IPv4) и с использованием TCP-соединений. В
    # бесконечном цикле ждем новых подключений. Для корректного завершения с помощью CTRL+C перехватываем исключение
    # KeyboardInterrupt и завершаем все потоки с подключенными сокетами.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # текущий сокет может переиспользовать адрес.
        server.bind(('127.0.0.1', 8000))
        server.listen()
        connection_threads = []
        try:
            while True:
                connection, addr = server.accept()
                thread = ClientEchoThread(connection)
                connection_threads.append(thread)
                thread.start()
        except KeyboardInterrupt:
            print('Останавливаюсь!')
            [thread.close() for thread in connection_threads]

"""
Для общения с вышеуказанным сокетом достаточно такой функции:

def client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Подключаемся к серверу на 127.0.0.1:8000
        sock.connect(('127.0.0.1', 8000))

        while True:
            body = input()
            if body == 'stop':
                break
            sock.sendall(body.encode('utf-8'))
            # Получаем ответ от сервера
            response = sock.recv(2048)
            print("Ответ от сервера:\n", response.decode('utf-8'))
if __name__ == '__main__':
    client()
"""