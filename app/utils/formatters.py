"""Formatting helpers for the new app-layer architecture."""

from pprint import pformat
from typing import Any


def pretty(value: Any) -> str:
    return pformat(value, sort_dicts=False)
