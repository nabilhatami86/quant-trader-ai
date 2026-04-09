from typing import Any
from app.api.schemas.common import SuccessResponse, ErrorResponse


def ok(message: str = "OK", data: Any = None) -> dict:
    return SuccessResponse(message=message, data=data).model_dump()


def err(message: str) -> dict:
    return ErrorResponse(error=message).model_dump()
