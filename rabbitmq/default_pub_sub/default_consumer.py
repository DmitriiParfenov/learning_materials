import datetime
import logging
import time
from typing import TYPE_CHECKING

from rabbitmq.config import (
    configure_logging,
    get_connection,
    settings
)

if TYPE_CHECKING:
    from pika.adapters.blocking_connection import BlockingChannel
    from pika.spec import Basic, BasicProperties

logger = logging.getLogger(__name__)


def process_new_messages(
        channel: 'BlockingChannel',
        method: 'Basic.Deliver',
        properties: 'BasicProperties',
        body: bytes,
):
    """
    Метод для обработки сообщений из очереди. Этот метод необходим согласно документации к получению (consuming)
    сообщений из RabbitMQ для работы метода channel.basic_consume(...)
    """
    logger.debug("channel: %s", channel)  # Канал, откуда берем сообщения.
    logger.debug("method: %s", method)
    logger.debug("properties: %s", properties)
    logger.debug("body: %s", body)

    # Начинаем обработку сообщения.
    logger.info("[ ] Start processing message %r", body)

    start_time = time.time()
    message = body.decode(encoding='utf-8')
    time.sleep(1)
    logger.info("Message from rabbitmq: %s", message)
    end_time = time.time()

    logger.info("Finished processing message %r, sending ack!", body)

    # Говорим RabbitMQ, что успешно обработали сообщение. ГАРАНТИЯ ДОСТАВКИ ИМЕЕНО ЗДЕСЬ И КРОЕТСЯ.
    channel.basic_ack(delivery_tag=method.delivery_tag)
    logger.info(
        "[X] Finished in %.2fs processing message %r",
        end_time - start_time,
        body,
    )


def main():
    """
    Метод создает подключение к RabbitMQ и потребляет сообщения.
    """

    configure_logging(level=logging.INFO)
    # Создаем подключение к RabbitMQ.
    with get_connection() as connection:
        logger.debug('Created connection: %s', connection)

        # Создаем канал в рамках одного подключения.
        with connection.channel() as channel:
            logger.debug('Created channel: %s', channel)

            # Объявляем очередь, откуда будем брать сообщения по ROUTING_KEY.
            custom_queue = channel.queue_declare(
                queue=settings.mq.routing_key,  # Имя очереди
                durable=True  # Очередь будет сохранена даже после перезагрузки RabbitMQ.
            )
            logger.debug("Declared queue %r %s", settings.mq.routing_key, custom_queue)

            # Настраиваем consumer на потребление сообщений из очереди по ROUTING_KEY.
            channel.basic_qos(prefetch_count=1)  # Каждый consumer потребляет по 1 задаче по мере загрузки.
            channel.basic_consume(
                queue=settings.mq.routing_key,  # Имя очереди
                on_message_callback=process_new_messages,  # Обработчик сообщения из RabbitMQ.
            )
            logger.info("Waiting to processing message")

            # Начинаем потреблять сообщение из RabbitMQ (блокирующий вызов).
            channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info('Connection is closing. Bye!')
