from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> bool:
    """
    Validasi API key dari header X-API-Key.
    - API_KEY kosong + API_KEY_REQUIRED=False → izinkan (dev mode)
    - API_KEY kosong + API_KEY_REQUIRED=True  → tolak semua request (misconfiguration)
    - API_KEY diset → wajib ada dan cocok
    """
    if not settings.API_KEY:
        if settings.API_KEY_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API key belum dikonfigurasi — set API_KEY di .env",
            )
        return True  # dev mode

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header X-API-Key wajib disertakan",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key tidak valid",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True
