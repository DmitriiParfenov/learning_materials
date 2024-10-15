import time
from concurrent import futures
from typing import NamedTuple


# Нет нужды импортировать multiprocessing, SimpleQueue и т. д., потому что concurrent.futures скрывает все это

NUMBERS = [
    9999999999999917,
    222222222222222222,
    142702110479723,
    231651651321651321,
    198465165416549651,
    4444444444444423,
    888888888888888888,
    498465165132165165,
    645132164198163516,
    311111111111111111,
    7777777777777753,
    299593572317531
]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


class PrimeResult(NamedTuple):
    n: int
    flag: bool
    elapsed: float


def check(n: int) -> PrimeResult:
    start = time.time()
    res = is_prime(n)
    return PrimeResult(n, res, time.time() - start)


def main() -> None:
    workers = 3

    executor = futures.ProcessPoolExecutor(workers)
    actual_workers = executor._max_workers  # type: ignore

    print(f'Checking {len(NUMBERS)} numbers with {actual_workers} processes:')

    start = time.time()
    numbers = sorted(NUMBERS, reverse=True)
    with executor:
        for n, prime, elapsed in executor.map(check, numbers):
            label = 'Yes' if prime else 'No'
            print(f'{n:16} {label} {elapsed:9.6f}s')

    end = time.time() - start
    print(f'Total time: {end:.2f}s')


if __name__ == '__main__':
    main()
