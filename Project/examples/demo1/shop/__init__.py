"""Tiny shop helpers with intentional logic bugs (for analysis demos)."""

from .auth import authorize, is_admin
from .billing import apply_discount, total_with_tax
from .cache import PriceCache
from .inventory import reserve_stock
from .paths import load_user_file

__all__ = [
    "authorize",
    "is_admin",
    "apply_discount",
    "total_with_tax",
    "PriceCache",
    "reserve_stock",
    "load_user_file",
]
