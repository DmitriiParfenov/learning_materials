import asyncio
import aiohttp


async def fetch_status(session: aiohttp.ClientSession, url: str) -> int:
    """
    Платформенная сопрограмма, которая делает запрос и возвращает статус код.
    """
    async with session.get(url) as result:
        return result.status


async def main():
    """
    Платформенная сопрограмма, которая создает сеанс. У сеанса может быть много подключений. Полный таймаут для всех
    подключений равен 1 сек, а для отдельно взятого - 0.1 сек. Для каждого подключения это значение можно переопределить
    в самом подключении (в данном случае в fetch_status).
    """
    session_timeout = aiohttp.ClientTimeout(total=1, connect=0.1)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        # Делаем запросы на сервисы в urls.
        urls = ['http://youtube.com/', 'python://youtube.com/']
        requests = [fetch_status(session, url) for url in urls]

        # Делаем это через gather, причем, если возникнет ошибка, то исключение не будет возбуждаться. Мы сможем
        # отделить исключения от успешных результатов.
        results = await asyncio.gather(*requests, return_exceptions=True)
        exceptions = [x for x in results if isinstance(x, Exception)]
        successful_results = [x for x in results if not isinstance(x, Exception)]

        print(f'Все результаты: {results}')
        print(f'Завершились успешно: {successful_results}')
        print(f'Завершились с исключением: {exceptions}')

if __name__ == '__main__':
    asyncio.run(main())
