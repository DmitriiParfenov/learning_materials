import asyncio
import random
from asyncio import Queue
from typing import Union, Sequence


class Product:
    """
    Класс Product инициализирует title - название продукта и time - время работа с товаром.
    """

    def __init__(self, title: str, time: Union[int, float]) -> None:
        self.title = title
        self.time = time


class Customer:
    """
    Класс Customer инициализирует name - имя клиента и products - корзину с продуктами.
    """

    def __init__(self, name: str, products: Sequence[Product]) -> None:
        self.name = name
        self.products = products


async def serve(queue: Queue) -> None:
    """
    Сопрограмма для работы с очередью.
    """
    # Работаем, пока в очереди есть объекты.
    while True:
        # Достаем из очереди объект и работаем с ним. Это блокирующая операция — пока нет доступных клиентов, ждем.
        customer: Customer = await queue.get()
        print(customer.name)
        for p in customer.products:
            print(f'Начинаю работать с {p.title}')
            await asyncio.sleep(p.time)
            print(f'Закончил работать с {p.title}')
        # Сообщаем, что работа с объектом закончена.
        queue.task_done()


async def main():
    # Объявляем очередь.
    main_queue = Queue(maxsize=2)
    # Объявляем продукты.
    products = [Product('пиво', 2), Product('мясо', 0.5), Product('мыло', 0.5), Product('сигареты', 5), ]
    # Объявляем покупателей.
    customers = [Customer('Dima', random.sample(products, 3)), Customer('Danya', random.sample(products, 3))]

    # Создаем 3 воркеров, которые будут работать с очередью.
    [asyncio.create_task(serve(main_queue)) for _ in range(3)]

    # Кладем объекты в очередь. Метод put – сопрограмма, при заполнении очереди main_queue произойдет блокировка
    # выполнения до появления свободного места.
    for i in customers:
        await main_queue.put(i)

    # Ждем заверешения работы со всеми объектами в очереди.
    await asyncio.gather(main_queue.join())


if __name__ == '__main__':
    asyncio.run(main())
