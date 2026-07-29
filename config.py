"""Central application settings, loaded from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "TradingBot"
    environment: str = "development"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    broker_api_key: str = ""
    broker_api_secret: str = ""
    broker_base_url: str = "https://paper-api.example.com"

    default_cash: float = 100_000.0
    default_commission: float = 0.001


@lru_cache
def get_settings() -> Settings:
    return Settings()
