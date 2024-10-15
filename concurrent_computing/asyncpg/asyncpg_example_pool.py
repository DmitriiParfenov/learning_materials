import asyncio
import os

import asyncpg
from dotenv import load_dotenv

dot_env = os.path.join('../..', '.env')
load_dotenv(dotenv_path=dot_env)

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS products (
        product_id SERIAL PRIMARY KEY,
        product_name VARCHAR(255) NOT NULL
    )
"""

INSERT_VALUE = """INSERT INTO products (product_name) VALUES ('apple'), ('orange'), ('tomato'), ('watermelon')"""


async def query_product(pool, query):
    """
    Сопрограмма забирает подключение из пула через pool.acquire() и останавливается до тех пор, пока не освободится
    подключения. Это делается в блоке async with, тем самым гарантируется, что при выходе из блока подключение будет
    возвращено в пул.
    """
    async with pool.acquire() as connection:
        return await connection.fetch(query)


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
        # Запрос на создание таблицы и внесение туда данных (синхронно).
        async with pool.acquire() as connection:
            await connection.execute(CREATE_TABLE)
            await connection.execute(INSERT_VALUE)  # то

        # Запросы, которые будут выполнены конкурентно.
        query_1 = 'SELECT product_id, product_name FROM products'
        query_2 = 'SELECT product_id, product_name FROM products'
        tasks = [query_product(pool, x) for x in [query_1, query_2]]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        result = [x for x in responses if not isinstance(x, Exception)]
        for q in result:
            for obj in q:
                print(obj.get('product_id'), obj.get('product_name'))
            print()


if __name__ == '__main__':
    asyncio.run(main())
