import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time


def is_prime(n: int) -> bool:
    print(f'Hello from {is_prime.__name__!r}')
    time.sleep(1)
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def is_even(n: int) -> bool:
    print(f'Hello from {is_even.__name__!r}')
    time.sleep(1)
    return n % 2 == 0


def main():
    start = time.perf_counter()
    tasks = [is_even, is_prime]

    with ProcessPoolExecutor(max_workers=2) as pool:
        to_do = [pool.submit(x, 60) for x in tasks]
        for task in concurrent.futures.as_completed(to_do):
            try:
                result = task.result()
                print(result)
            except Exception as e:
                print(e)

    elapsed = time.perf_counter() - start
    print(f'Время исполнения скрипта: {elapsed:.2f}')


if __name__ == '__main__':
    main()
