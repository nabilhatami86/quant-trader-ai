from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import MLResult


async def insert_ml_result(db: AsyncSession, data: dict) -> MLResult:
    ml = MLResult(**data)
    db.add(ml)
    await db.commit()
    await db.refresh(ml)
    return ml


async def get_latest_ml_result(
    db: AsyncSession, symbol: str, timeframe: str
) -> MLResult | None:
    result = await db.execute(
        select(MLResult)
        .where(MLResult.symbol == symbol, MLResult.timeframe == timeframe)
        .order_by(MLResult.trained_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def build_ml_payload(symbol: str, timeframe: str, train_result: dict) -> dict:
    """Konversi hasil bot.train_model() ke dict model MLResult."""
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_type": train_result.get("model_type", "ensemble"),
        "accuracy": train_result.get("accuracy", 0.0),
        "conf_accuracy": train_result.get("conf_accuracy", 0.0),
        "precision_buy": train_result.get("precision_buy", 0.0),
        "recall_buy": train_result.get("recall_buy", 0.0),
        "f1_score": train_result.get("f1", 0.0),
        "n_features": train_result.get("n_features", 0),
        "n_train": train_result.get("n_train", 0),
        "n_test": train_result.get("n_test", 0),
        "n_sideways_removed": train_result.get("n_sideways_removed", 0),
    }
