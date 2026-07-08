from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ALPACA_",
        extra="ignore",
    )

    api_key: str = Field(...)
    api_secret: str = Field(...)
    paper: bool = Field(default=True)


class SharadarSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHARADAR_",
        extra="ignore",
    )

    us_bundle_api_key: str = Field(...)
