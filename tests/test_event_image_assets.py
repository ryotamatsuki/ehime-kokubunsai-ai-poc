from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import event_image_assets as image_assets
from event_image_assets import (
    EVENT_IMAGE_ASSET_DIR,
    EVENT_IMAGE_MANIFEST_PATH,
    _is_valid_png_bytes,
    event_image_path,
    resolve_event_image,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        len(data).to_bytes(4, "big")
        + chunk_type
        + data
        + (zlib.crc32(chunk_type + data) & 0xFFFFFFFF).to_bytes(4, "big")
    )


def _valid_png(
    *,
    width: int = 1,
    height: int = 1,
    raw_scanlines: bytes = b"\x00\xff\x00\x00\xff",
    idat_data: bytes | None = None,
    include_iend: bool = True,
    corrupt_idat_crc: bool = False,
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    compressed = zlib.compress(raw_scanlines) if idat_data is None else idat_data
    idat = _png_chunk(b"IDAT", compressed)
    if corrupt_idat_crc:
        idat = idat[:-1] + bytes([idat[-1] ^ 0x01])
    iend = _png_chunk(b"IEND", b"") if include_iend else b""
    return PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + idat + iend


def test_manifest_and_asset_dir_are_local_paths() -> None:
    assert isinstance(EVENT_IMAGE_MANIFEST_PATH, Path)
    assert isinstance(EVENT_IMAGE_ASSET_DIR, Path)
    assert EVENT_IMAGE_MANIFEST_PATH.is_file()
    assert EVENT_IMAGE_ASSET_DIR.is_dir()


def test_all_manifest_events_resolve_to_strict_valid_local_pngs() -> None:
    manifest = json.loads(EVENT_IMAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    event_mappings = manifest["events"]

    assert len(event_mappings) == 30
    assert set(event_mappings) == {f"{index:03d}" for index in range(1, 31)}

    for event_id, mapping in event_mappings.items():
        path = event_image_path(event_id)
        assert path is not None
        assert path == (EVENT_IMAGE_ASSET_DIR / mapping["file"]).resolve()
        assert path.parent == EVENT_IMAGE_ASSET_DIR.resolve()
        payload = path.read_bytes()
        assert payload.startswith(PNG_SIGNATURE)
        assert _is_valid_png_bytes(payload)


def test_numeric_ids_and_alias_function_resolve_the_same_asset() -> None:
    assert resolve_event_image(1) == resolve_event_image("001")
    assert event_image_path("001") == resolve_event_image("001")


def test_unknown_or_unusable_ids_return_none() -> None:
    for event_id in (None, "", "unknown", "999", -1, True, 1.5):
        assert resolve_event_image(event_id) is None


def test_missing_and_invalid_manifest_mappings_return_none(
    tmp_path: Path, monkeypatch
) -> None:
    asset_root = tmp_path / "assets" / "events"
    asset_root.mkdir(parents=True)
    (asset_root / "valid.png").write_bytes(_valid_png())
    (asset_root / "invalid.png").write_bytes(b"not-a-png")
    (tmp_path / "outside.png").write_bytes(PNG_SIGNATURE + b"outside")

    manifest_path = tmp_path / "event_images.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "events": {
                    "001": {"file": "valid.png", "status": "ready"},
                    "002": {"file": "missing.png", "status": "ready"},
                    "003": {"file": "valid.jpg", "status": "ready"},
                    "004": {"file": "../outside.png", "status": "ready"},
                    "005": {"file": "valid.png", "status": "pending"},
                    "006": {"file": None, "status": "ready"},
                    "007": "not-a-mapping",
                    "008": {"file": "invalid.png", "status": "ready"},
                    "009": {
                        "file": str((tmp_path / "outside.png").resolve()),
                        "status": "ready",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_ASSET_DIR", asset_root)

    assert resolve_event_image("001") == (asset_root / "valid.png").resolve()
    for event_id in ("002", "003", "004", "005", "006", "007", "008", "009"):
        assert resolve_event_image(event_id) is None


def test_malformed_pngs_return_none_but_valid_png_is_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    asset_root = tmp_path / "assets" / "events"
    asset_root.mkdir(parents=True)

    valid_payload = _valid_png()
    malformed_payloads = {
        "002": valid_payload[:-1],
        "003": _valid_png(corrupt_idat_crc=True),
        "004": _valid_png(include_iend=False),
        "005": _valid_png(idat_data=b"not-a-zlib-stream"),
        "006": _valid_png(width=0),
        "007": _valid_png(raw_scanlines=b"\x00"),
    }
    (asset_root / "valid.png").write_bytes(valid_payload)
    for event_id, payload in malformed_payloads.items():
        (asset_root / f"event_{event_id}.png").write_bytes(payload)

    manifest_path = tmp_path / "event_images.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "events": {
                    "001": {"file": "valid.png", "status": "ready"},
                    **{
                        event_id: {
                            "file": f"event_{event_id}.png",
                            "status": "ready",
                        }
                        for event_id in malformed_payloads
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_ASSET_DIR", asset_root)

    assert event_image_path("001") == (asset_root / "valid.png").resolve()
    for event_id in malformed_payloads:
        assert event_image_path(event_id) is None


def test_missing_or_malformed_manifest_returns_none(
    tmp_path: Path, monkeypatch
) -> None:
    missing_manifest = tmp_path / "missing.json"
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_MANIFEST_PATH", missing_manifest)
    assert resolve_event_image("001") is None

    malformed_manifest = tmp_path / "malformed.json"
    malformed_manifest.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(image_assets, "EVENT_IMAGE_MANIFEST_PATH", malformed_manifest)
    assert resolve_event_image("001") is None
