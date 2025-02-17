import datetime
import json
import logging

import pika

from rabbitmq.config import (
    configure_logging,
    get_connection,
    settings
)

logger = logging.getLogger(__name__)


def main():
    """
    Метод создает подключение к RabbitMQ и публикует сообщения.
    """

    configure_logging()
    # Создаем подключение к RabbitMQ.
    with get_connection() as connection:
        logger.debug('Created connection: %s', connection)

        # Создаем канал в рамках одного подключения.
        with connection.channel() as channel:
            logger.debug('Created channel: %s', channel)

            # Объявляем очередь, куда будем слать сообщения по ROUTING_KEY.
            custom_queue = channel.queue_declare(
                queue=settings.mq.routing_key,  # Имя очереди
                durable=True  # Очередь будет сохранена даже после перезагрузки RabbitMQ.
            )
            logger.debug("Declared queue %r %s", settings.mq.routing_key, custom_queue)

            # Формируем сообщение.
            message = f'Hello from queue {settings.mq.routing_key!r}: time - {datetime.datetime.utcnow()}'
            message_body = message.encode(encoding='utf-8')
            logger.debug("Created message %s", message_body)

            # Публикуем сообщение.
            channel.basic_publish(
                exchange=settings.mq.exchange,
                routing_key=settings.mq.routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Сообщение будет сохранено даже после перезагрузки RabbitMQ.
                    headers={'task_queue': f'{settings.mq.routing_key}'}
                )
            )
            logger.info("Published message %s", message_body)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info('Connection is closing. Bye!')
