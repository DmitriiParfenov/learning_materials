import os
import time
from queue import Queue, Empty
from threading import Thread
from dotenv import load_dotenv
import requests

from parallel_programming.token_bucket import Throttle

THREAD_POOL_SIZE = 4
SYMBOLS = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')
BASES = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')
throttle = Throttle(10)
dot_env = os.path.join('..', '.env')
load_dotenv(dotenv_path=dot_env)


def fetch_rates(base):
    """Функция для получения курса валют для base."""

    # Делаем запрос на сайт для получения курса валют
    url = f"https://api.apilayer.com/exchangerates_data/latest?base={base}"
    payload = {}
    headers = {
        "apikey": os.getenv('API_KEY')
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    response.raise_for_status()
    # Для base получаем все возможные курсы валют, которые дает сайт
    rates = response.json()["rates"]
    # Примечание: валюта обменивается сама на себя с коэффициентом 1:1
    rates[base] = 1.
    # Получает текущий курс валют в отношении symbols и выводим в консоль.
    return base, rates


def worker(work_queue, results_queue):
    """Извлекает из очереди задачи и выполняет их. Метод task_done() уведомляет о завершении задачи."""

    # Цикл идет, пока в очереди есть хоть одна задача.
    while not work_queue.empty():
        # Пробуем вытаскивать из очереди задачу, если ее нет, то мы не блокируем очередь, а выходим из цикла while.
        # Если удалось вытащить задачу, то для нее выполняем функцию fetch_rates() и уведомляем о выполнении.
        try:
            item = work_queue.get(block=False)
        except Empty:
            break
        else:
            while not throttle.consume():
                pass
            # Добавили results_queue для того, чтобы принтовать в консоль. Если на запрос API пришла ошибка, то
            # обрабатываем ее.
            try:
                result = fetch_rates(item)
            except Exception as err:
                results_queue.put(err)
            else:
                results_queue.put(result)
            finally:
                work_queue.task_done()


def present_result(base, rates):
    rates_line = ", ".join([f"{rates[symbol]:7.03} {symbol}" for symbol in SYMBOLS])
    print(f"1 {base} = {rates_line}")


def main():
    # Создаем очередь
    work_queue = Queue()
    results_queue = Queue()

    # Добавляем в очередь каждую валюту для конвертации.
    for base in BASES:
        work_queue.put(base)

    # Определяем все потоки в количестве, равном THREAD_POOL_SIZE для контроля количества потоков
    threads = [
        Thread(target=worker, args=(work_queue, results_queue))
        for _ in range(THREAD_POOL_SIZE)
    ]

    # Сразу запускаем все потоки.
    for thread in threads:
        thread.start()

    #  Блокируем выполнение программы до тех пор, пока все элементы в очереди не будут обработаны и извлечены.
    #  Если в момент вызова метода в очереди есть элементы, программа останавливается и ждет, пока все элементы не будут
    #  извлечены. Только после этого выполнение программы продолжится дальше
    work_queue.join()

    while threads:
        threads.pop().join()

    while not results_queue.empty():
        result = results_queue.get()
        if isinstance(result, Exception):
            raise result

        present_result(*result)


if __name__ == "__main__":
    started = time.time()
    main()
    elapsed = time.time() - started
    print()
    print("time elapsed: {:.2f}s".format(elapsed))
