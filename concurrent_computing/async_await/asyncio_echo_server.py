import asyncio
from asyncio import StreamReader, StreamWriter


class ServerState:
    """
    Класс, который позволяет запускать эхо сервер с использованием API asyncio.
    """

    async def start_server(self, host: str, port: int):
        """
        Сопрограмма создает сервер. Через asyncio.start_server мы получаем AbstractServer, у которого запускаем метод
        serve_forever для того, чтобы исполнялся код сервера до момента его остановки.
        self.client_connected - это сопрограмма обратного вызова, которая вызывается, когда к серверу подключается
        клиент: принимает StreamReader и StreamWriter.
        """
        server = await asyncio.start_server(self.client_connected, host, port)  # получили AbstractServer.
        async with server:  # запустили сервер так, чтобы можно было корректно завершить его работу
            await server.serve_forever()

    async def client_connected(self, reader: StreamReader, writer: StreamWriter):
        """
        Сопрограмма для общения с клиентами.
        """
        addr = writer.get_extra_info('peername')
        print(f'Подключен клиент {addr}')
        try:
            while True:
                data = await reader.read(2048)
                if not data:
                    break
                message = data.decode()
                print(f'Получено сообщение от {addr}: {message}')
                # Отправляем данные клиенту.
                writer.write(bytes('Hello!', encoding='utf-8'))
                await writer.drain()
            print(f'Клиент {addr} отключился')
        except Exception as e:
            print(f'Ошибка у клиента {addr}: {e}')
        finally:
            writer.close()
            await writer.wait_closed()


async def main():
    chat_server = ServerState()
    await chat_server.start_server('127.0.0.1', 8000)
    print('Сервер запущен на 127.0.0.1:8000')


if __name__ == '__main__':
    asyncio.run(main())


"""
Код Клиента

import asyncio

async def tcp_client():
    reader, writer = await asyncio.open_connection('127.0.0.1', 8000)

    while True:
        message = input("Введите сообщение для отправки на сервер: ")
        if message == 'stop':
            break
        writer.write(message.encode())
        await writer.drain()
        data = await reader.read(2048)
        print(f'Ответ от сервера: {data.decode()}')

    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(tcp_client())
"""