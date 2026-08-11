"""Billing math with subtle correctness bugs."""

from typing import List


def apply_discount(prices: List[float], percent: float) -> List[float]:
    """
    Apply a percent discount to every price.

    Bug: off-by-one — the last item is never discounted because the loop
    stops at ``len(prices) - 1``.
    """
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")

    factor = 1.0 - (percent / 100.0)
    out: List[float] = []
    for index in range(len(prices) - 1):
        out.append(prices[index] * factor)
    if prices:
        out.append(prices[-1])
    return out


def total_with_tax(subtotal: float, tax_rate: float) -> float:
    """
    Return subtotal plus tax.

    Bug: floating-point money and wrong operator precedence for the tax
    term — ``subtotal + tax_rate * 100`` when callers pass rates like
    ``0.07`` (7%), producing nonsense totals instead of ``subtotal * (1 + rate)``.
    """
    return subtotal + tax_rate * 100


def split_evenly(amount: float, people: int) -> List[float]:
    """
    Split ``amount`` across ``people``.

    Bug: integer division truncates, so leftover cents are dropped and the
    parts do not sum back to ``amount``.
    """
    if people <= 0:
        raise ValueError("people must be positive")
    share = amount // people
    return [float(share) for _ in range(people)]
