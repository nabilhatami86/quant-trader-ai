from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────
    APP_NAME: str = "Trader AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Security ──────────────────────────────────────────────────
    API_KEY: str = ""  # kosong = tanpa auth (dev mode)

    # ── Database (baca dari .env) ─────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "forex_trading"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── Bot defaults ──────────────────────────────────────────────
    BOT_SYMBOL: str = "XAUUSD"
    BOT_TIMEFRAME: str = "1h"
    BOT_USE_MT5: bool = False
    BOT_USE_NEWS: bool = True

    # ── MetaTrader 5 connection ────────────────────────────────────
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""

    # ── Trading params (dipakai saat BOT_USE_MT5=true) ────────────
    BOT_LOT: float = 0.01           # lot size per order
    BOT_ORDERS: int = 1             # jumlah order sekaligus
    BOT_TRAIL_PIPS: float = 0.0     # trailing stop (0 = off)
    BOT_DCA_MINUTES: float = 0.0    # DCA interval (0 = off)
    BOT_REAL_MODE: bool = False     # mode akun real (auto-lot, multi-TP)
    BOT_MICRO_MODE: bool = False    # mode akun mikro (filter ketat)
    BOT_RISK_USD: float = 0.0       # max risk per order USD (0 = pakai ATR)

    # ── CORS ──────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
