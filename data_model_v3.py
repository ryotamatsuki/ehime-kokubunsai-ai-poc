"""Validated Data Model v3 overlay for the cultural-event PoC.

The existing ``data/events.json`` remains the v2 compatibility source.  v3
normalizes P0 facts in two sidecars and composes them on read. Missing
coordinates and last-admission times remain explicitly unknown; this module
never infers them from names, venue types, or an LLM.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EVENTS_V2_PATH = DATA_DIR / "events.json"
EVENT_PROFILES_V3_PATH = DATA_DIR / "event_profiles_v3.json"
VENUES_V3_PATH = DATA_DIR / "venues_v3.json"
SCHEMA_VERSION = 3

EVENT_STATUS_VALUES = frozenset({"scheduled", "postponed", "rescheduled", "cancelled", "completed"})
POSTURE_VALUES = frozenset({"mostly_seated", "mixed", "standing_or_walking", "unknown"})
SEATING_VALUES = frozenset({"guaranteed", "available", "limited", "none", "unknown"})
MOBILITY_LOAD_VALUES = frozenset({"low", "medium", "high", "unknown"})
ENGAGEMENT_MODE_VALUES = frozenset({"watch", "listen", "hands_on", "audience_participation", "walk_explore"})
DURATION_BASIS_VALUES = frozenset({"scheduled_program", "poc_authored", "official", "unknown"})
LAST_ADMISSION_STATUS_VALUES = frozenset({"known", "unknown", "not_applicable"})
LOCATION_TYPE_VALUES = frozenset({"facility", "facility_and_surroundings", "area", "route", "synthetic"})
ADDRESS_STATUS_VALUES = frozenset({"verified", "display_text_only", "unknown"})
GEO_PRECISION_VALUES = frozenset({"entrance", "venue_exact", "area_anchor", "municipality_anchor", "unknown"})
GEO_ENRICHMENT_STATUS_VALUES = frozenset({"verified", "pending_verification", "not_applicable"})
PROVENANCE_SOURCE_TYPES = frozenset({"official", "organizer", "poc_authored", "human_verified", "other"})
PROVENANCE_DERIVATIONS = frozenset({"explicit", "normalized", "rule_derived", "human_verified", "poc_definition", "copied_from_v2", "llm_inferred"})
_TIME_RE = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d")

_PROFILE_POLICY = {
    "event_status": "poc_definition",
    "experience_profile": "poc_definition",
    "estimated_visit_duration": "basis_driven",
    "last_admission": "status_driven",
    "llm_inferred_hard_filter": False,
}
_VENUE_POLICY = {
    "identity": "copied_from_v2",
    "address": "copied_from_v2",
    "geo": "explicit_unknown_until_verified",
    "llm_inferred_routing": False,
}


class DataModelV3Error(ValueError):
    pass


def _fail(message: str, path: str) -> None:
    raise DataModelV3Error(f"{path}: {message}")


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _obj(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("must be an object", path)
    return value


def _text(value: Any, path: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail("must be a non-empty string", path)
    return value.strip()


def _enum(value: Any, allowed: frozenset[str], path: str) -> str:
    text = _text(value, path)
    assert text is not None
    if text not in allowed:
        _fail(f"must be one of {sorted(allowed)}", path)
    return text


def _exact(raw: Mapping[str, Any], keys: set[str], path: str) -> None:
    if set(raw) != keys:
        _fail(f"must contain exactly {sorted(keys)}", path)


def validate_fact_provenance(value: Any, *, path: str = "provenance") -> dict[str, Any]:
    raw = _obj(value, path)
    _exact(raw, {"source_type", "source_ref", "derivation", "hard_filter_eligible", "note"}, path)
    source_type = _enum(raw["source_type"], PROVENANCE_SOURCE_TYPES, f"{path}.source_type")
    source_ref = _text(raw["source_ref"], f"{path}.source_ref")
    derivation = _enum(raw["derivation"], PROVENANCE_DERIVATIONS, f"{path}.derivation")
    hard = raw["hard_filter_eligible"]
    if not isinstance(hard, bool):
        _fail("must be a boolean", f"{path}.hard_filter_eligible")
    note = _text(raw["note"], f"{path}.note", nullable=True)
    if derivation == "llm_inferred" and hard:
        _fail("LLM-inferred facts must never be hard-filter eligible", path)
    return {"source_type": source_type, "source_ref": source_ref, "derivation": derivation, "hard_filter_eligible": hard, "note": note}


def validate_event_profile_v3(value: Any, *, path: str = "event_profile") -> dict[str, Any]:
    raw = _obj(value, path)
    _exact(raw, {"event_id", "event_status", "venue_id", "experience_profile", "estimated_visit_duration", "last_admission"}, path)
    event_id = _text(raw["event_id"], f"{path}.event_id")
    assert event_id is not None
    if re.fullmatch(r"\d{3}", event_id) is None:
        _fail("must be a three-digit event id", f"{path}.event_id")
    event_status = _enum(raw["event_status"], EVENT_STATUS_VALUES, f"{path}.event_status")
    venue_id = _text(raw["venue_id"], f"{path}.venue_id")
    assert venue_id is not None

    exp = _obj(raw["experience_profile"], f"{path}.experience_profile")
    _exact(exp, {"posture", "seating", "mobility_load", "engagement_modes"}, f"{path}.experience_profile")
    modes = exp["engagement_modes"]
    if not isinstance(modes, list) or not modes or len(modes) != len(set(modes)):
        _fail("must be a non-empty duplicate-free array", f"{path}.experience_profile.engagement_modes")
    experience = {
        "posture": _enum(exp["posture"], POSTURE_VALUES, f"{path}.experience_profile.posture"),
        "seating": _enum(exp["seating"], SEATING_VALUES, f"{path}.experience_profile.seating"),
        "mobility_load": _enum(exp["mobility_load"], MOBILITY_LOAD_VALUES, f"{path}.experience_profile.mobility_load"),
        "engagement_modes": [_enum(v, ENGAGEMENT_MODE_VALUES, f"{path}.experience_profile.engagement_modes") for v in modes],
    }

    duration = _obj(raw["estimated_visit_duration"], f"{path}.estimated_visit_duration")
    _exact(duration, {"typical_minutes", "basis"}, f"{path}.estimated_visit_duration")
    minutes = duration["typical_minutes"]
    if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 1440):
        _fail("typical_minutes must be null or 1..1440", f"{path}.estimated_visit_duration")
    basis = _enum(duration["basis"], DURATION_BASIS_VALUES, f"{path}.estimated_visit_duration.basis")
    if (basis == "unknown") != (minutes is None):
        _fail("unknown basis iff typical_minutes is null", f"{path}.estimated_visit_duration")

    last = _obj(raw["last_admission"], f"{path}.last_admission")
    _exact(last, {"time", "status"}, f"{path}.last_admission")
    status = _enum(last["status"], LAST_ADMISSION_STATUS_VALUES, f"{path}.last_admission.status")
    last_time = _text(last["time"], f"{path}.last_admission.time", nullable=True)
    if status == "known" and (last_time is None or _TIME_RE.fullmatch(last_time) is None):
        _fail("known status requires HH:MM time", f"{path}.last_admission")
    if status != "known" and last_time is not None:
        _fail("unknown/not_applicable requires time=null", f"{path}.last_admission")

    return {"event_id": event_id, "event_status": event_status, "venue_id": venue_id, "experience_profile": experience, "estimated_visit_duration": {"typical_minutes": minutes, "basis": basis}, "last_admission": {"time": last_time, "status": status}}


def load_event_profiles_v3(path: str | Path = EVENT_PROFILES_V3_PATH) -> dict[str, dict[str, Any]]:
    raw = _obj(_load(path), "event_profiles_v3")
    _exact(raw, {"schema_version", "dataset", "description", "provenance_policy", "events"}, "event_profiles_v3")
    if raw["schema_version"] != SCHEMA_VERSION or raw["provenance_policy"] != _PROFILE_POLICY:
        _fail("schema_version or provenance_policy mismatch", "event_profiles_v3")
    if not isinstance(raw["events"], list):
        _fail("events must be an array", "event_profiles_v3.events")
    result: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(raw["events"]):
        profile = validate_event_profile_v3(item, path=f"event_profiles_v3.events[{i}]")
        if profile["event_id"] in result:
            _fail("duplicate event_id", "event_profiles_v3.events")
        result[profile["event_id"]] = profile
    return result


def validate_venue_v3(value: Any, *, path: str = "venue") -> dict[str, Any]:
    raw = _obj(value, path)
    _exact(raw, {"venue_id", "name", "municipality", "region", "location_type", "address_text", "address_status", "geo"}, path)
    venue_id = _text(raw["venue_id"], f"{path}.venue_id")
    name = _text(raw["name"], f"{path}.name")
    municipality = _text(raw["municipality"], f"{path}.municipality")
    region = _text(raw["region"], f"{path}.region")
    location_type = _enum(raw["location_type"], LOCATION_TYPE_VALUES, f"{path}.location_type")
    address_text = _text(raw["address_text"], f"{path}.address_text", nullable=True)
    address_status = _enum(raw["address_status"], ADDRESS_STATUS_VALUES, f"{path}.address_status")
    if address_status == "verified" and address_text is None:
        _fail("verified address requires address_text", path)

    geo = _obj(raw["geo"], f"{path}.geo")
    _exact(geo, {"latitude", "longitude", "precision", "routing_eligible", "enrichment_status"}, f"{path}.geo")
    lat, lon = geo["latitude"], geo["longitude"]
    for v, low, high, name_ in ((lat, -90.0, 90.0, "latitude"), (lon, -180.0, 180.0, "longitude")):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not low <= float(v) <= high):
            _fail(f"invalid {name_}", f"{path}.geo.{name_}")
    if (lat is None) != (lon is None):
        _fail("latitude and longitude must both be null or both be present", f"{path}.geo")
    precision = _enum(geo["precision"], GEO_PRECISION_VALUES, f"{path}.geo.precision")
    routing = geo["routing_eligible"]
    if not isinstance(routing, bool):
        _fail("must be a boolean", f"{path}.geo.routing_eligible")
    enrichment = _enum(geo["enrichment_status"], GEO_ENRICHMENT_STATUS_VALUES, f"{path}.geo.enrichment_status")
    if lat is None and (precision != "unknown" or routing):
        _fail("missing coordinates require precision=unknown and routing_eligible=false", f"{path}.geo")
    if routing and (precision not in {"entrance", "venue_exact"} or enrichment != "verified"):
        _fail("routing requires verified venue-level coordinates", f"{path}.geo")
    return {"venue_id": venue_id, "name": name, "municipality": municipality, "region": region, "location_type": location_type, "address_text": address_text, "address_status": address_status, "geo": {"latitude": None if lat is None else float(lat), "longitude": None if lon is None else float(lon), "precision": precision, "routing_eligible": routing, "enrichment_status": enrichment}}


def load_venues_v3(path: str | Path = VENUES_V3_PATH) -> dict[str, dict[str, Any]]:
    raw = _obj(_load(path), "venues_v3")
    _exact(raw, {"schema_version", "dataset", "description", "provenance_policy", "venues"}, "venues_v3")
    if raw["schema_version"] != SCHEMA_VERSION or raw["provenance_policy"] != _VENUE_POLICY:
        _fail("schema_version or provenance_policy mismatch", "venues_v3")
    if not isinstance(raw["venues"], list):
        _fail("venues must be an array", "venues_v3.venues")
    result: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(raw["venues"]):
        venue = validate_venue_v3(item, path=f"venues_v3.venues[{i}]")
        if venue["venue_id"] in result:
            _fail("duplicate venue_id", "venues_v3.venues")
        result[venue["venue_id"]] = venue
    return result


def _prov(source_ref: str, derivation: str, hard: bool, note: str) -> dict[str, Any]:
    return validate_fact_provenance({"source_type": "poc_authored", "source_ref": source_ref, "derivation": derivation, "hard_filter_eligible": hard, "note": note})


def _profile_provenance(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    eid = str(profile["event_id"])
    basis = profile["estimated_visit_duration"]["basis"]
    last_status = profile["last_admission"]["status"]
    return {
        "event_status": _prov(f"data/event_profiles_v3.json#{eid}.event_status", "poc_definition", True, "PoC架空イベントの開催状態。"),
        "experience_profile": _prov(f"data/event_profiles_v3.json#{eid}.experience_profile", "poc_definition", True, "LLM推測ではなくPoC仕様として定義した体験特性。"),
        "estimated_visit_duration": _prov(
            f"data/events.json#{eid}.start_datetime,end_datetime" if basis == "scheduled_program" else f"data/event_profiles_v3.json#{eid}.estimated_visit_duration",
            "rule_derived" if basis == "scheduled_program" else "poc_definition",
            basis != "unknown",
            "固定プログラムは開始・終了時刻から算出。随時入場はPoC典型滞在時間。",
        ),
        "last_admission": _prov(
            f"data/events.json#{eid}.参加形式" if last_status == "not_applicable" else f"data/event_profiles_v3.json#{eid}.last_admission",
            "rule_derived" if last_status == "not_applicable" else "poc_definition",
            last_status in {"known", "not_applicable"},
            "未登録の最終入場時刻は終了時刻等から推測しない。",
        ),
    }


def _venue_provenance(venue: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    vid = str(venue["venue_id"])
    ref = f"data/venues_v3.json#{vid}"
    return {
        "identity": _prov(ref, "copied_from_v2", False, "v2の場所表記を会場IDへ正規化。"),
        "address": _prov(ref + ".address_text", "copied_from_v2", False, "表示テキストであり郵便住所として未検証。"),
        "geo": _prov(ref + ".geo", "poc_definition", False, "根拠がない座標はnull/unknownのまま保持。"),
    }


def compose_events_v3(events_v2: Iterable[Mapping[str, Any]], profiles: Mapping[str, Mapping[str, Any]], venues: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    events = [dict(v) for v in events_v2]
    if len(events) != 30:
        _fail("expected exactly 30 v2 events", "events_v2")
    ids = [str(e.get("id", "")) for e in events]
    if len(ids) != len(set(ids)) or any(re.fullmatch(r"\d{3}", eid) is None for eid in ids):
        _fail("event ids must be unique three-digit strings", "events_v2")
    if set(ids) != set(profiles):
        _fail(f"profile id mismatch; missing={sorted(set(ids)-set(profiles))}, extra={sorted(set(profiles)-set(ids))}", "event_profiles_v3")
    result: list[dict[str, Any]] = []
    for event in events:
        eid = str(event["id"])
        profile = validate_event_profile_v3(profiles[eid], path=f"profile[{eid}]")
        vid = profile["venue_id"]
        if vid not in venues:
            _fail(f"unknown venue_id {vid}", f"profile[{eid}].venue_id")
        venue = validate_venue_v3(venues[vid], path=f"venue[{vid}]")
        if str(event.get("市町", "")) != venue["municipality"] or str(event.get("地域", "")) != venue["region"]:
            _fail("event municipality/region differs from venue master", f"event[{eid}]")
        enriched = deepcopy(event)
        enriched.update({"data_model_version": 3, "event_status": profile["event_status"], "venue_id": vid, "venue_v3": {**deepcopy(venue), "provenance": _venue_provenance(venue)}, "experience_profile": deepcopy(profile["experience_profile"]), "estimated_visit_duration": deepcopy(profile["estimated_visit_duration"]), "last_admission": deepcopy(profile["last_admission"]), "provenance_v3": _profile_provenance(profile)})
        result.append(enriched)
    return result


def load_events_v3(events_path: str | Path = EVENTS_V2_PATH, profiles_path: str | Path = EVENT_PROFILES_V3_PATH, venues_path: str | Path = VENUES_V3_PATH) -> list[dict[str, Any]]:
    events = _load(events_path)
    if not isinstance(events, list):
        _fail("must be an array", "events_v2")
    return compose_events_v3(events, load_event_profiles_v3(profiles_path), load_venues_v3(venues_path))


def hard_filter_eligible(event_v3: Mapping[str, Any], field_name: str = "experience_profile") -> bool:
    if field_name not in {"event_status", "experience_profile", "estimated_visit_duration", "last_admission"}:
        _fail("unknown provenance field", "field_name")
    provenance = _obj(event_v3.get("provenance_v3"), "event.provenance_v3")
    fact = validate_fact_provenance(provenance.get(field_name), path=f"event.provenance_v3.{field_name}")
    return fact["hard_filter_eligible"] and fact["derivation"] != "llm_inferred"


__all__ = ["DataModelV3Error", "EVENT_PROFILES_V3_PATH", "EVENTS_V2_PATH", "SCHEMA_VERSION", "VENUES_V3_PATH", "compose_events_v3", "hard_filter_eligible", "load_event_profiles_v3", "load_events_v3", "load_venues_v3", "validate_event_profile_v3", "validate_fact_provenance", "validate_venue_v3"]
