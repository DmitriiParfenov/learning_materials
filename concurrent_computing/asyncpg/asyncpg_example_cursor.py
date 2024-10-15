import asyncio
import os

import asyncpg
from dotenv import load_dotenv

dot_env = os.path.join('../..', '.env')
load_dotenv(dotenv_path=dot_env)


async def main():
    """
    Создали пул из 6 подключений, то есть конкурентно можно сделать 6 SQL-запросов (1 подключение = 1 запрос).
    """
    async with asyncpg.create_pool(host='127.0.0.1',
                                   port=5432,
                                   user='postgres',
                                   database='products',
                                   password=os.environ.get('password'),
                                   min_size=6,
                                   max_size=6) as pool:
        query = 'SELECT * FROM products'
        async with pool.acquire() as connection:
            async with connection.transaction():
                async for x in connection.cursor(query):
                    print(x.get('product_name'))


if __name__ == '__main__':
    asyncio.run(main())
