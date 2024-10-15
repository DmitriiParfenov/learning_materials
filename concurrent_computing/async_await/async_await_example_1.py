import asyncio
import time

import aiohttp

SYMBOLS = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')
BASES = ('USD', 'EUR', 'PLN', 'NOK', 'CZK')


# Объявляем сопрограмму для получения курсов валют от API.
async def get_rates(session: aiohttp.ClientSession, base: str):
    headers = {
        "apikey": "wOPYR2f5u1jMYugxyvSrSlpRUsF0HpSb"
    }

    # Отправляем GET запрос к API для получения курсов валют по указанной базовой валюте
    async with session.get(f"https://api.apilayer.com/exchangerates_data/latest?base={base}",
                           headers=headers) as response:
        # Получаем курсы валют, причем ждем результат текущей задачи, не блокируя основной поток выполнения.
        rates = (await response.json())['rates']
        rates[base] = 1.
    return base, rates


# Объявляем сопрограмму для асинхронного получения курсов валют.
async def fetch_rates(session, place):
    return await get_rates(session, place)


# Объявляем сопрограмму для отображения результатов полученных курсов валют.
async def present_result(result):
    base, rates = (await result)
    rates_line = ", ".join(
        [f"{rates[symbol]:7.03} {symbol}" for symbol in SYMBOLS]
    )
    print(f"1 {base} = {rates_line}")


# Объявляем сопрограмму для выполнения основной логики программы.
async def main():
    # Создание сессии для отправки HTTP запросов.
    async with aiohttp.ClientSession() as session:
        # Асинхронное ожидание завершения всех запросов курсов валют.
        await asyncio.gather(*(present_result(fetch_rates(session, base)) for base in BASES))


if __name__ == "__main__":
    started = time.time()
    asyncio.run(main())
    elapsed = time.time() - started
    print()
    print("time elapsed: {:.2f}s".format(elapsed))
