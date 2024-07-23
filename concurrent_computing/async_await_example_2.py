import asyncio
import time
from pathlib import Path
from typing import Callable

from httpx import AsyncClient

"""
1) Сначала функция main запускает функцию download_many.
2) download_many запускает цикл событий, который приводит в действие объект сопрограммы supervisor до того момента, пока 
supervisor не вернет управление.
3) supervisor создает список сопрограмм и для каждого объекта вызывает функцию supervisor для асинхронного обращения 
к удаленному серверу, передавая supervisor экземпляр класса AsyncClient и сам объект сопрограммы. supervisor работает 
до тех пор, пока не получит все результаты объектов, допускающих ожидание (должны реализовать __await__).
4) download_one загружает изображение флага для заданного кода страны и сохраняет его на диск.
"""

# Объявление переменных.
POP20_CC = 'CN IN US ID BR PK NG BD RU JP MX PH VN ET EG DE IR TR CD FR'.split()  # Флаги таких стран получим.
BASE_URL = 'https://www.fluentpython.com/data/flags'  # Откуда будем скачивать изображения флагов указанных стран.
DEST_DIR = Path('downloaded')  # В директорию downloaded будет складывать изображения флагов стран.


def save_flag(img: bytes, filename: str) -> None:
    """
    Метод сохраняет изображение, представленное в виде байтового потока, в файл.
    """
    (DEST_DIR / filename).write_bytes(img)


async def get_flag(client: AsyncClient, cc: str) -> bytes:
    """
    Метод получает изображение флага для заданного кода страны.
    Метод обращается к удаленному серверу для получения изображения флага,
    соответствующего заданному коду страны, и возвращает его в виде байтового потока.
    """
    url = f'{BASE_URL}/{cc}/{cc}.gif'.lower()
    resp = await client.get(url, timeout=6.1, follow_redirects=True)
    return resp.read()


async def download_one(client: AsyncClient, cc: str):
    """
    Метод загружает изображение флага для заданного кода страны и сохраняет его на диск.
    """
    image = await get_flag(client, cc)
    save_flag(image, f'{cc}.gif')
    print(cc, end=' ', flush=True)
    return cc


def download_many(cc_list: list[str]) -> int:
    """
    Функция запускает цикл событий, который приводит в действие объект сопрограммы supervisor до того момента, пока
    supervisor не вернет управление.
    """
    return asyncio.run(supervisor(cc_list))


async def supervisor(cc_list: list[str]) -> int:
    """
    Платформенная сопрограмма, которая асинхронно применяет функцию download_one для каждого объекта списка cc_list,
    передавая экземпляр класса AsyncClient и сам объект в качестве аргументов. Работает до тех пор, пока все объекты,
    ждущих завершения не будут выполнены.
    """
    # AsyncClient - асинхронные менеджер контекста.
    async with AsyncClient() as client:
        # to_do - это список сопрограмм.
        to_do = [download_one(client, cc) for cc in sorted(cc_list)]
        # gather управляет всеми сопрограммами в коде. Также можно через цикл for obj in asyncio.as_completed(to_do).
        res = await asyncio.gather(*to_do)
    return len(res)


def main(downloader: Callable[[list[str]], int]) -> None:
    DEST_DIR.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    count = downloader(POP20_CC)
    elapsed = time.perf_counter() - t0
    print(f'\n{count} downloads in {elapsed:.2f}s')


if __name__ == '__main__':
    main(download_many)
