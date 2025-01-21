import asyncio
import hashlib
import os
import random
import string
import time


# 41,54 - синхр
#

def random_password(length: int) -> bytes:
    """
    Метод генерирует случайный пароль длиной length.
    """
    ascii_lowercase = string.ascii_lowercase.encode()
    return b''.join(bytes(random.choice(ascii_lowercase)) for _ in range(length))


def hash(password: bytes) -> str:
    """
    Метод хеширует указанный пароль.
    """
    salt = os.urandom(16)
    return str(hashlib.scrypt(password, salt=salt, n=2048, p=1, r=8))


async def main():
    """
    Сопрограмма, которая хеширует пароля при помощи hashlib с использованием многопоточности. Метод scrypt библиотека
    hashlib не захватывает GIL. Выигрыш в сравнее с синхронной версией примерно 5 раз.
    """
    start = time.time()
    passwords = [random_password(10) for _ in range(10000)]     # Получаем пароли.
    tasks = [asyncio.to_thread(hash, p) for p in passwords]     # Хешируем пароли в режиме многопоточности.
    results = await asyncio.gather(*tasks)
    print(results)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    asyncio.run(main())
