from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> bool:
    """
    Validasi API key dari header X-API-Key.
    Jika API_KEY di .env kosong, semua request diperbolehkan (dev mode).
    """
    if not settings.API_KEY:
        return True

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True
