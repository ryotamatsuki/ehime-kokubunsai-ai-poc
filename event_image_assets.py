"""Resolve event-card image assets without importing the UI layer."""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parent

EVENT_IMAGE_MANIFEST_PATH = _REPOSITORY_ROOT / "data" / "event_images.manifest.json"
EVENT_IMAGE_ASSET_DIR = _REPOSITORY_ROOT / "assets" / "events"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR = b"IHDR"
_PNG_IDAT = b"IDAT"
_PNG_IEND = b"IEND"

_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)

__all__ = [
    "EVENT_IMAGE_ASSET_DIR",
    "EVENT_IMAGE_MANIFEST_PATH",
    "event_image_path",
    "resolve_event_image",
]


def _normalize_event_id(event_id: object) -> str | None:
    """Return the manifest's three-digit event ID, or ``None`` if unusable."""

    if isinstance(event_id, bool):
        return None
    if isinstance(event_id, int):
        if event_id < 0:
            return None
        return f"{event_id:03d}"
    if not isinstance(event_id, str):
        return None

    normalized = event_id.strip()
    if not normalized or not normalized.isdigit():
        return None
    return normalized.zfill(3)


def _read_manifest() -> Mapping[str, Any] | None:
    try:
        manifest_path = Path(EVENT_IMAGE_MANIFEST_PATH)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError):
        return None

    return manifest if isinstance(manifest, Mapping) else None


def _is_chunk_type(chunk_type: bytes) -> bool:
    return len(chunk_type) == 4 and all(
        65 <= byte <= 90 or 97 <= byte <= 122 for byte in chunk_type
    )


def _parse_ihdr(data: bytes) -> tuple[int, int, int, int, int] | None:
    if len(data) != 13:
        return None

    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", data)
    )
    if width <= 0 or height <= 0:
        return None
    if compression != 0 or filter_method != 0 or interlace not in (0, 1):
        return None

    allowed_bit_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if bit_depth not in allowed_bit_depths.get(color_type, set()):
        return None

    channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_by_color_type[color_type]
    return width, height, bit_depth, channels, interlace


def _pass_extent(size: int, start: int, step: int) -> int:
    if size <= start:
        return 0
    return (size - start + step - 1) // step


def _validate_scanlines(
    decoded: bytes,
    width: int,
    height: int,
    bit_depth: int,
    channels: int,
    interlace: int,
) -> bool:
    bits_per_pixel = bit_depth * channels

    if interlace == 0:
        row_bytes = (width * bits_per_pixel + 7) // 8
        row_length = 1 + row_bytes
        if len(decoded) != height * row_length:
            return False
        return all(decoded[offset] <= 4 for offset in range(0, len(decoded), row_length))

    row_specs: list[tuple[int, int]] = []
    for x_start, y_start, x_step, y_step in _ADAM7_PASSES:
        pass_width = _pass_extent(width, x_start, x_step)
        pass_height = _pass_extent(height, y_start, y_step)
        if pass_width == 0 or pass_height == 0:
            continue
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        row_specs.append((1 + row_bytes, pass_height))

    expected_length = sum(row_length * row_count for row_length, row_count in row_specs)
    if len(decoded) != expected_length:
        return False

    offset = 0
    for row_length, row_count in row_specs:
        for _ in range(row_count):
            if decoded[offset] > 4:
                return False
            offset += row_length
    return offset == len(decoded)


def _is_valid_png_bytes(payload: bytes) -> bool:
    """Validate PNG structure and compressed image data using stdlib only."""

    if not payload.startswith(_PNG_SIGNATURE):
        return False

    position = len(_PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    idat_closed = False
    iend_seen = False

    while position < len(payload):
        remaining = len(payload) - position
        if remaining < 12:
            return False

        data_length = int.from_bytes(payload[position : position + 4], "big")
        position += 4
        chunk_type = payload[position : position + 4]
        position += 4
        if not _is_chunk_type(chunk_type):
            return False

        data_end = position + data_length
        crc_end = data_end + 4
        if data_end < position or crc_end > len(payload):
            return False

        chunk_data = payload[position:data_end]
        stored_crc = int.from_bytes(payload[data_end:crc_end], "big")
        calculated_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if stored_crc != calculated_crc:
            return False
        position = crc_end

        if ihdr is None:
            if chunk_type != _PNG_IHDR:
                return False
            ihdr = _parse_ihdr(chunk_data)
            if ihdr is None:
                return False
            continue

        if chunk_type == _PNG_IHDR:
            return False
        if chunk_type == _PNG_IDAT:
            if idat_closed:
                return False
            idat_parts.append(chunk_data)
            continue

        if idat_parts:
            idat_closed = True
        if chunk_type == _PNG_IEND:
            if chunk_data or not idat_parts or iend_seen:
                return False
            iend_seen = True
            if position != len(payload):
                return False
            break

    if ihdr is None or not idat_parts or not iend_seen:
        return False

    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(b"".join(idat_parts))
        decoded += decompressor.flush()
    except zlib.error:
        return False

    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        return False

    width, height, bit_depth, channels, interlace = ihdr
    return _validate_scanlines(decoded, width, height, bit_depth, channels, interlace)


@lru_cache(maxsize=128)
def _is_valid_png_file(path_string: str, file_size: int, modified_ns: int) -> bool:
    """Validate an immutable local asset at most once per file version."""

    try:
        return _is_valid_png_bytes(Path(path_string).read_bytes())
    except OSError:
        return False


def _safe_png_path(filename: object) -> Path | None:
    if not isinstance(filename, str):
        return None

    filename = filename.strip()
    if not filename:
        return None

    try:
        relative_filename = Path(filename)
        if relative_filename.is_absolute():
            return None
        root = Path(EVENT_IMAGE_ASSET_DIR).resolve()
        candidate = (root / relative_filename).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    if candidate.suffix.casefold() != ".png" or not candidate.is_file():
        return None

    try:
        file_stat = candidate.stat()
    except OSError:
        return None

    return (
        candidate
        if _is_valid_png_file(str(candidate), file_stat.st_size, file_stat.st_mtime_ns)
        else None
    )


def event_image_path(event_id: object) -> Path | None:
    """Resolve an event ID to a verified local PNG asset.

    The manifest is the source of the filename mapping. Any unknown ID,
    malformed mapping, unsafe path, unavailable file, or non-PNG file returns
    ``None`` so callers can safely omit the optional image.
    """

    normalized_event_id = _normalize_event_id(event_id)
    if normalized_event_id is None:
        return None

    manifest = _read_manifest()
    if manifest is None:
        return None

    events = manifest.get("events")
    if not isinstance(events, Mapping):
        return None

    mapping = events.get(normalized_event_id)
    if not isinstance(mapping, Mapping):
        return None

    status = mapping.get("status")
    if status is not None and status != "ready":
        return None

    return _safe_png_path(mapping.get("file"))


def resolve_event_image(event_id: object) -> Path | None:
    """Backward-friendly name for :func:`event_image_path`."""

    return event_image_path(event_id)
