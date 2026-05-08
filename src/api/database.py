from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class DatabaseSettings:
    url: str | None

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def safe_url(self) -> str | None:
        if not self.url:
            return None

        parsed = urlsplit(self.url)
        if "@" not in parsed.netloc:
            return self.url

        credentials, host = parsed.netloc.rsplit("@", 1)
        user = credentials.split(":", 1)[0]
        return urlunsplit(parsed._replace(netloc=f"{user}:***@{host}"))


def load_database_settings() -> DatabaseSettings:
    return DatabaseSettings(url=os.getenv("DATABASE_URL"))


def database_status(settings: DatabaseSettings | None = None) -> dict:
    db_settings = settings or load_database_settings()
    if not db_settings.configured:
        return {
            "name": "database",
            "status": "not_configured",
            "detail": "DATABASE_URL is not set; using local processed CSV artifacts.",
            "metadata": {"url": None},
        }

    return {
        "name": "database",
        "status": "configured",
        "detail": "DATABASE_URL is configured for the live TimescaleDB service.",
        "metadata": {"url": db_settings.safe_url},
    }
