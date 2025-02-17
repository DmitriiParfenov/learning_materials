import logging

import pika
from rabbitmq.config import settings

# Определяем параметры для подключения к брокеру Rabbit.
connection_params = pika.ConnectionParameters(
    host=settings.mq.host,
    port=settings.mq.port,
    credentials=pika.PlainCredentials(
        username=settings.mq.user,
        password=settings.mq.password
    )
)


# Создаем подключение к брокеру RabbitMQ.
def get_connection() -> pika.BlockingConnection:
    """
    Метод для создания подключения к брокеру RabbitMQ.
    """
    return pika.BlockingConnection(
        parameters=connection_params
    )


# Метод для настроек логирования.
def configure_logging(level: int = logging.INFO):
    """
    Определяем настройки для логирования.
    """
    logging.basicConfig(
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    # format="[%(asctime)s.%(msecs)03d] %(levelname)-3s [%(module)-20s] %(funcName)-35s [%(lineno)-4d] - %(message)s",
        format="[%(asctime)s.%(msecs)03d] %(levelname)-3s: %(message)s"
    )
