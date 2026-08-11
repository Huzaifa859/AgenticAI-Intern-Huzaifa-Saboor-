"""In-memory price cache with a keying bug."""

from typing import Dict, Optional, Tuple


class PriceCache:
    """Cache catalog prices per (sku, currency)."""

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}

    def _key(self, sku: str, currency: str) -> str:
        """
        Build a cache key.

        Bug: currency is ignored, so USD and EUR entries overwrite each
        other under the same sku.
        """
        return sku

    def set_price(self, sku: str, currency: str, price: float) -> None:
        self._store[self._key(sku, currency)] = price

    def get_price(self, sku: str, currency: str) -> Optional[float]:
        return self._store.get(self._key(sku, currency))

    def bulk_get(self, items: Tuple[str, str]) -> Optional[float]:
        """
        Look up one price from a (sku, currency) pair.

        Bug: argument order is documented as (sku, currency) but the
        implementation treats the tuple as (currency, sku), returning the
        wrong entry whenever both differ.
        """
        currency, sku = items
        return self.get_price(sku, currency)
