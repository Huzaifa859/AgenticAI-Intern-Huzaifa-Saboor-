"""Stock reservation with check-then-act and boundary bugs."""

from typing import Dict


def reserve_stock(inventory: Dict[str, int], sku: str, qty: int) -> bool:
    """
    Reserve ``qty`` units of ``sku`` from ``inventory``.

    Bugs:
    1. Uses ``>`` instead of ``>=``, so reserving exactly the remaining
       stock always fails even though enough units exist.
    2. Check-then-act: reads ``available``, then mutates later without
       re-checking, so a second caller can oversell if this runs twice
       on the same dict between check and decrement.
    """
    if qty <= 0:
        return False

    available = inventory.get(sku, 0)
    if available > qty:
        inventory[sku] = available - qty
        return True
    return False


def paginate(items: list, page: int, page_size: int) -> list:
    """
    Return one page of ``items`` (1-based page index).

    Bug: end index uses ``page * page_size - 1``, dropping the last item
    of every page (classic off-by-one slice).
    """
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be >= 1")
    start = (page - 1) * page_size
    end = page * page_size - 1
    return items[start:end]
