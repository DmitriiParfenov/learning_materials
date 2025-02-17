from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class RabbitMQSettings(BaseSettings):
    host: str = '127.0.0.1'
    port: int = 5672
    user: str = 'guest'
    password: str = 'guest'
    exchange: str = ''
    routing_key: str = 'default'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        env_file=BASE_DIR / '.env',
        env_prefix='APP__',
        case_sensitive=False,
    )
    mq: RabbitMQSettings = RabbitMQSettings()


settings = Settings()
