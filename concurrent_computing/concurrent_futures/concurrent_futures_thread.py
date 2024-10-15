import time
from concurrent import futures
from pathlib import Path
from typing import Callable

import httpx

# Объявление переменных.
POP20_CC = 'CN IN US ID BR PK NG BD RU JP MX PH VN ET EG DE IR TR CD FR QWERT'.split()  # Флаги таких стран получим.
BASE_URL = 'https://www.fluentpython.com/data/flags'  # Откуда будем скачивать изображения флагов указанных стран.
DEST_DIR = Path('downloaded')  # В директорию downloaded будет складывать изображения флагов стран.


def save_flag(img: bytes, filename: str) -> None:
    """
    Метод сохраняет изображение, представленное в виде байтового потока, в файл.
    """
    (DEST_DIR / filename).write_bytes(img)


def get_flag(cc: str) -> bytes:
    """
    Метод получает изображение флага для заданного кода страны.
    Метод обращается к удаленному серверу для получения изображения флага,
    соответствующего заданному коду страны, и возвращает его в виде байтового потока.
    """
    url = f'{BASE_URL}/{cc}/{cc}.gif'.lower()
    resp = httpx.get(url, timeout=6.1, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def download_one(cc: str):
    """
    Метод загружает изображение флага для заданного кода страны и сохраняет его в файл.
    """
    image = get_flag(cc)
    save_flag(image, f'{cc}.gif')
    print(cc, end=' ', flush=True)
    return cc


def download_many_1(cc_list: list[str]) -> int:
    """
    Метод загружает изображения флагов для списка кодов стран с использованием многопоточности.
    Метод создает пул потоков и запускает параллельное скачивание изображений флагов
    для каждого кода страны в списке. После скачивания изображения сохраняются в файлы.
    """
    # Создали экземпляр ThreadPoolExecutor как контекстный менеджер; метод executor.__exit__ вызовет
    # executor.shutdown(wait=True), который блокирует выполнение программы до завершения всех потоков.
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        # res - это результат выполнения будущих объектов.
        res = executor.map(download_one, sorted(cc_list))
    return len(list(res))


def download_many_2(cc_list: list[str]) -> int:
    """
    Аналогично функции download_many_1, но выводит в терминал список запланированных фьючерсов (будущих объектов)
    и результат их выполнения.
    """
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        # executor.submit планирует выполнение будущих объектов и возвращает объект future, который мы кладем в to_do.
        to_do: dict[futures.Future, str] = dict()
        for cc in sorted(cc_list):
            future = executor.submit(download_one, cc)
            to_do[future] = cc
            print(f'Scheduled for {cc}: {future}')

        # futures.as_completed(to_do) принимает итерируемый объект с будущими объектами и возвращает итератор, который
        # отдает будущие объекты по мере их выполнения.
        for count, future in enumerate(futures.as_completed(to_do), 1):
            try:
                status = future.result()
                print(f'{future} result: {status!r}')
            except httpx.HTTPStatusError as exc:
                error_msg = 'HTTP error {resp.status_code} - {resp.reason_phrase}'
                error_msg = error_msg.format(resp=exc.response)
            except httpx.RequestError as exc:
                error_msg = f'{exc} {type(exc)}'.strip()
            except KeyboardInterrupt:
                break
            else:
                error_msg = ''
            if error_msg:
                flag = to_do[future]
                print(f'{future} error: {error_msg} {flag}')
    return count


def main(downloader: Callable[[list[str]], int]) -> None:
    DEST_DIR.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    count = downloader(POP20_CC)
    elapsed = time.perf_counter() - t0
    print(f'\n{count} downloads in {elapsed:.2f}s')


if __name__ == '__main__':
    main(download_many_2)
