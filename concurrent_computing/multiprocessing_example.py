import os
import time
from multiprocessing import Pool

import requests
from dotenv import load_dotenv

SYMBOLS = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')
BASES = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')
POOL_SIZE = 4
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


def present_result(base, rates):
    rates_line = ", ".join([f"{rates[symbol]:7.03} {symbol}" for symbol in SYMBOLS])
    print(f"1 {base} = {rates_line}")


def main():
    with Pool(POOL_SIZE) as pool:
        results = pool.map(fetch_rates, BASES)

    for result in results:
        present_result(*result)


if __name__ == "__main__":
    started = time.time()
    main()
    elapsed = time.time() - started
    print()
    print("time elapsed: {:.2f}s".format(elapsed))
