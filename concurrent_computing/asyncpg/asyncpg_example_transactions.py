import asyncio
import os

import asyncpg
from dotenv import load_dotenv

dot_env = os.path.join('../..', '.env')
load_dotenv(dotenv_path=dot_env)


async def query_product_1(pool, query):
    """
    Сопрограмма забирает подключение из пула через pool.acquire() и останавливается до тех пор, пока не освободится
    подключения. Это делается в блоке async with, тем самым гарантируется, что при выходе из блока подключение будет
    возвращено в пул.
    """
    async with pool.acquire() as connection:
        try:
            async with connection.transaction():
                await asyncio.sleep(3)
                return await connection.fetch(query)
        except Exception as er:
            print(er)


async def query_product_2(pool, query):
    """
    Для ручного управления транзакциями.
    """
    async with pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await asyncio.sleep(3)
            await connection.execute(query)
        except asyncpg.PostgresError:
            await transaction.rollball()
        else:
            await transaction.commit()


async def main():
    """
    Создали пул из 6 подключений, то есть конкурентно можно сделать 6 SQL-запросов (1 подключение = 1 запрос).
    Сопрограмма query_product забирает подключение из пула через pool.acquire() и останавливается до тех пор, пока не
    освободится подключения. Это делается в блоке async with; тем самым гарантируется, что при выходе из блока
    подключение будет возвращено в пул.
    """
    async with asyncpg.create_pool(host='127.0.0.1',
                                   port=5432,
                                   user='postgres',
                                   database='products',
                                   password=os.environ.get('password'),
                                   min_size=6,
                                   max_size=6) as pool:
        # Запросы, которые будут выполнены конкурентно (с ошибкой).
        start = asyncio.get_event_loop().time()
        query_1 = 'CREATE TABLE IF NOT EXISTS products'
        query_2 = "INSERT INTO products (product_id, product_name) VALUES (1, 'apple')"
        query_3 = "INSERT INTO products (product_id, product_name) VALUES (1, 'orange')"

        # Запускаем три запроса, которые будут обернуты в транзакции.
        tasks = [query_product_1(pool, x) for x in [query_1, query_2, query_3]]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Запускаем запрос, который покажет нам данные в таблице products.
        query_for_show = """SELECT * FROM products"""
        result = await query_product_1(pool, query_for_show)
        print([(x.get('product_id'), x.get('product_name')) for x in result])
        end = asyncio.get_event_loop().time()
        print(f'Программа завершилась за {end - start}')


if __name__ == '__main__':
    asyncio.run(main())
