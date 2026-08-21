"""Pure helpers for bounded, client-side result pagination."""

from __future__ import annotations

from typing import Sequence, TypeVar

from app_config import RESULT_PAGE_SIZE


T = TypeVar("T")


def normalize_visible_count(
    total_count: int,
    visible_count: int | None,
    *,
    page_size: int = RESULT_PAGE_SIZE,
) -> int:
    """Clamp a stored page count without ever hiding the first page."""

    total = max(0, int(total_count))
    page = max(1, int(page_size))
    if total == 0:
        return 0
    if visible_count is None:
        return min(page, total)
    return min(total, max(page, int(visible_count)))


def next_visible_count(
    total_count: int,
    visible_count: int | None,
    *,
    page_size: int = RESULT_PAGE_SIZE,
) -> int:
    """Return the next client-side page boundary for a result set."""

    current = normalize_visible_count(total_count, visible_count, page_size=page_size)
    return min(max(0, int(total_count)), current + max(1, int(page_size)))


def visible_items(
    items: Sequence[T],
    visible_count: int | None,
    *,
    page_size: int = RESULT_PAGE_SIZE,
) -> list[T]:
    """Slice an already ordered result set without re-ranking or de-duping it."""

    count = normalize_visible_count(len(items), visible_count, page_size=page_size)
    return list(items[:count])
