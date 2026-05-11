import functools
import logging

from botocore.exceptions import ClientError, ConnectionError, BotoCoreError, EndpointConnectionError

logger = logging.getLogger(__name__)


def s3_error_handler(default=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, EndpointConnectionError):
                logger.warning("Не удалось подключиться к S3.")
                return default
            except ClientError as error:
                logger.warning("Ошибка при работе с S3: %s", str(error))
                return default
            except BotoCoreError as error:
                logger.warning("Ошибка при работе с S3: %s", str(error))
                return default

        return wrapper

    return decorator
