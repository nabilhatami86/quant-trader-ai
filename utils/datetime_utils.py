from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def fmt(dt: datetime | str | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    return dt.strftime(fmt)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
