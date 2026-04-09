from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

CACHE_DIR = DATA_DIR / "cache"
HISTORY_DIR = DATA_DIR / "history"
JOURNAL_DIR = DATA_DIR / "journal"
SESSION_DIR = DATA_DIR / "session"

TRADE_JOURNAL_PATH = JOURNAL_DIR / "trade_journal.csv"
SESSION_BIAS_STATE_PATH = SESSION_DIR / "session_bias_state.json"
SESSION_PLAN_PATH = SESSION_DIR / "session_plan.json"

# Keep backward compat — adaptive state lives in ai/adaptive_state.json at project root
ADAPTIVE_STATE = PROJECT_ROOT / 'ai' / 'adaptive_state.json'


def ensure_runtime_dirs() -> None:
    for directory in (
        DATA_DIR,
        CACHE_DIR,
        HISTORY_DIR,
        JOURNAL_DIR,
        SESSION_DIR,
        LOGS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()
