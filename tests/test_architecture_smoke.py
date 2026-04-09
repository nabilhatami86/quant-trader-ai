import importlib


MODULES = [
    "app",
    "app.api",
    "app.api.deps",
    "app.api.routes.signal",
    "app.api.schemas.trade",
    "app.core.config",
    "app.core.security",
    "app.database.session",
    "app.database.models",
    "app.database.crud.trades",
    "app.engine.bot",
    "app.engine.broker.mt5_connector",
    "app.engine.signals.signals",
    "app.services.bot_service",
    "app.services.db_logger",
    "app.services.ai.adaptive",
    "app.services.ai.ml.predictor",
    "app.services.news.news_filter",
    "app.services.tradingview.tv_feed",
    "app.utils.response",
]


def test_app_modules_importable():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, f"failed to import {module_name}"


def test_legacy_wrappers_still_importable():
    legacy_modules = [
        "api.deps",
        "backend.bot",
        "backend.broker.mt5_connector",
        "db.database",
        "db.models",
        "services.bot_service",
        "utils.response",
    ]
    for module_name in legacy_modules:
        module = importlib.import_module(module_name)
        assert module is not None, f"failed to import legacy wrapper {module_name}"


def test_runtime_paths_exist():
    from app.core.paths import CACHE_DIR, JOURNAL_DIR, SESSION_DIR, TRADE_JOURNAL_PATH

    assert CACHE_DIR.exists()
    assert JOURNAL_DIR.exists()
    assert SESSION_DIR.exists()
    assert TRADE_JOURNAL_PATH.parent == JOURNAL_DIR
