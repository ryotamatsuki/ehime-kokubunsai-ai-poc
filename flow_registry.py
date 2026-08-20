"""Single source of truth for semantic command flows.

``CommandPlan`` carries only a flow name.  This registry is the trusted
mapping from that name to a short description, required slots, and a fixed
executor identifier.  Callers should dispatch through this table rather than
executing a tool name supplied by an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from command_models import COMMAND_SLOT_FIELDS, FLOW_NAMES


class FlowRegistryError(ValueError):
    """Raised when a flow specification or registry is malformed."""


def _fail(message: str, path: str | None = None) -> None:
    rendered = f"{path}: {message}" if path else message
    raise FlowRegistryError(rendered)


def _text(value: Any, *, path: str, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("must be a non-empty string", path)
    normalized = value.strip()
    if "\x00" in normalized or any(ord(character) < 32 for character in normalized):
        _fail("must not contain control characters", path)
    if len(normalized) > max_length:
        _fail(f"must be at most {max_length} characters", path)
    return normalized


@dataclass(frozen=True)
class FlowSpec:
    """Trusted metadata for one semantic flow."""

    name: str
    description: str
    required_slots: tuple[str, ...] = ()
    executor_name: str = "none"

    def __post_init__(self) -> None:
        name = _text(self.name, path="flow.name", max_length=64)
        if name not in FLOW_NAMES:
            _fail(f"unknown flow {name!r}", "flow.name")
        description = _text(self.description, path=f"flow_registry.{name}.description")

        if not isinstance(self.required_slots, (list, tuple)):
            _fail("must be an array/tuple", f"flow_registry.{name}.required_slots")
        if len(self.required_slots) > len(COMMAND_SLOT_FIELDS):
            _fail("contains too many required slots", f"flow_registry.{name}.required_slots")
        required: list[str] = []
        for index, slot_name in enumerate(self.required_slots):
            slot = _text(
                slot_name,
                path=f"flow_registry.{name}.required_slots[{index}]",
                max_length=64,
            )
            if slot not in COMMAND_SLOT_FIELDS:
                _fail(f"unknown command slot {slot!r}", f"flow_registry.{name}.required_slots[{index}]")
            if slot in required:
                _fail(f"duplicate required slot {slot!r}", f"flow_registry.{name}.required_slots")
            required.append(slot)

        executor_name = _text(
            self.executor_name,
            path=f"flow_registry.{name}.executor_name",
            max_length=64,
        )
        if re.fullmatch(r"[a-z][a-z0-9_]*", executor_name) is None:
            _fail("must be a safe executor identifier", f"flow_registry.{name}.executor_name")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "required_slots", tuple(required))
        object.__setattr__(self, "executor_name", executor_name)

    @classmethod
    def from_dict(cls, raw: Any) -> "FlowSpec":
        """Create a FlowSpec from a JSON-like object with no extra fields."""

        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("must be an object", "flow")
        allowed = frozenset({"name", "description", "required_slots", "executor_name"})
        unknown = [key for key in raw if key not in allowed]
        if unknown:
            _fail(f"unknown field(s): {sorted(map(str, unknown))}", "flow")
        missing = [key for key in ("name", "description", "executor_name") if key not in raw]
        if missing:
            _fail(f"missing required field(s): {missing}", "flow")
        return cls(
            name=raw["name"],
            description=raw["description"],
            required_slots=raw.get("required_slots", ()),
            executor_name=raw["executor_name"],
        )

    def validate(self) -> "FlowSpec":
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required_slots": list(self.required_slots),
            "executor_name": self.executor_name,
        }


FLOW_REGISTRY: dict[str, FlowSpec] = {
    "find_events": FlowSpec(
        name="find_events",
        description=(
            "利用者の場所、日時、同行者、年齢、料金、"
            "興味・テーマなどの希望に合う文化祭イベントを探す"
        ),
        required_slots=(),
        executor_name="search_events",
    ),
    "count_events": FlowSpec(
        name="count_events",
        description="利用者が指定した条件に合うイベントの件数を知る",
        required_slots=(),
        executor_name="count_events",
    ),
    "event_detail": FlowSpec(
        name="event_detail",
        description=(
            "特定イベントの日時、場所、料金、申込、"
            "対象、アクセス等の事実を確認する"
        ),
        required_slots=(),
        executor_name="get_event_detail",
    ),
    "recommend_next": FlowSpec(
        name="recommend_next",
        description="あるイベントの終了後、同じ日に続けて参加可能な別イベントを探す",
        required_slots=(),
        executor_name="recommend_next_events",
    ),
    "recommend_similar": FlowSpec(
        name="recommend_similar",
        description="選んだイベントと内容やジャンルが近い別イベントを探す",
        required_slots=(),
        executor_name="recommend_similar_events",
    ),
    "plan_event_pair": FlowSpec(
        name="plan_event_pair",
        description=(
            "同じ日に2つのイベントを回る、はしごする、"
            "午前と午後に1つずつ参加する等の複数イベント参加の組み合わせを探す"
        ),
        required_slots=("dates",),
        executor_name="recommend_event_pairs",
    ),
    "general_faq": FlowSpec(
        name="general_faq",
        description="特定イベントではなく文化祭全体についてよくある質問を確認する",
        required_slots=(),
        executor_name="search_faq",
    ),
    "unsupported": FlowSpec(
        name="unsupported",
        description="このPoCのイベント検索・参加案内の範囲外",
        required_slots=(),
        executor_name="none",
    ),
}


def validate_flow_registry(
    registry: Mapping[str, FlowSpec] | None = None,
) -> Mapping[str, FlowSpec]:
    """Validate registry topology and return the checked mapping."""

    target = FLOW_REGISTRY if registry is None else registry
    if not isinstance(target, Mapping):
        _fail("must be a mapping", "FLOW_REGISTRY")
    if set(target) != set(FLOW_NAMES):
        missing = sorted(set(FLOW_NAMES) - set(target))
        extra = sorted(set(target) - set(FLOW_NAMES))
        _fail(f"flow names do not match; missing={missing}, extra={extra}", "FLOW_REGISTRY")
    for name, spec in target.items():
        if not isinstance(name, str) or name not in FLOW_NAMES:
            _fail(f"unknown registry key {name!r}", "FLOW_REGISTRY")
        if not isinstance(spec, FlowSpec):
            _fail("value must be FlowSpec", f"FLOW_REGISTRY.{name}")
        if spec.name != name:
            _fail(f"spec name must equal registry key {name!r}", f"FLOW_REGISTRY.{name}.name")
    return target


validate_flow_registry()


def get_flow_spec(flow: str) -> FlowSpec:
    """Return a registered flow or raise a clear validation error."""

    if not isinstance(flow, str):
        _fail("must be a string", "flow")
    try:
        return FLOW_REGISTRY[flow]
    except KeyError:
        _fail(f"unknown flow {flow!r}", "flow")
    raise AssertionError("unreachable")  # pragma: no cover


def executor_name_for(flow: str) -> str:
    """Return the trusted executor identifier for a semantic flow."""

    return get_flow_spec(flow).executor_name


def required_slots_for(flow: str) -> tuple[str, ...]:
    """Return the slots Python must request before executing ``flow``."""

    return get_flow_spec(flow).required_slots


def flow_descriptions(
    registry: Mapping[str, FlowSpec] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return ordered ``(flow, description)`` pairs for prompt construction."""

    target = FLOW_REGISTRY if registry is None else validate_flow_registry(registry)
    return tuple((name, target[name].description) for name in FLOW_REGISTRY if name in target)


def render_flow_descriptions(
    registry: Mapping[str, FlowSpec] | None = None,
) -> str:
    """Render the registry descriptions without maintaining a second prompt list."""

    return "\n".join(f"- {name}: {description}" for name, description in flow_descriptions(registry))


__all__ = [
    "FLOW_NAMES",
    "FLOW_REGISTRY",
    "FlowRegistryError",
    "FlowSpec",
    "executor_name_for",
    "flow_descriptions",
    "get_flow_spec",
    "render_flow_descriptions",
    "required_slots_for",
    "validate_flow_registry",
]
