import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
    ]

    for h in handlers:
        h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Mute noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("trader_ai")


logger = setup_logging()
