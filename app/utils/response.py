from typing import Any
from app.api.schemas.common import SuccessResponse, ErrorResponse


def ok(message_or_data: Any = "OK", data: Any = None) -> dict:
    """Bisa dipanggil dua cara:
      ok("pesan", data)  → message="pesan", data=data
      ok({"key": val})   → message="OK",    data={"key": val}
    """
    if isinstance(message_or_data, str):
        message = message_or_data
    else:
        data    = message_or_data
        message = "OK"
    return SuccessResponse(message=message, data=data).model_dump()


def err(message: str) -> dict:
    return ErrorResponse(error=message).model_dump()
