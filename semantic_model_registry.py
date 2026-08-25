"""Model registry for the Semantic Operations v2.2 PoC.

The semantic architecture, prompt, schema, verifier, reducer and executor are
shared across models. Only the model backend changes. Keeping that boundary
explicit makes the Streamlit selector a genuine model A/B control rather than
an architecture switch.

Modal web-function URLs are public routing identifiers, not credentials. The
endpoints still require Modal proxy authentication, so the existing Modal key
and secret remain outside source control. Optional Streamlit URL secrets can
override these stable deployed URLs without changing the model contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_MODEL_KEY = "sarashina-2.2-3b"


@dataclass(frozen=True)
class SemanticModelSpec:
    key: str
    label: str
    short_label: str
    model_id: str
    endpoint_url: str
    modal_url_secret: str
    description: str


MODEL_SPECS: tuple[SemanticModelSpec, ...] = (
    SemanticModelSpec(
        key="sarashina-2.2-3b",
        label="Sarashina 2.2 3B",
        short_label="Sarashina 3B",
        model_id="sbintuitions/sarashina2.2-3b-instruct-v0.1",
        endpoint_url="https://ryota-matsuki--ehime-kokubunsai-semantic-v2-2-api-sarash-52fee3.modal.run",
        modal_url_secret="MODAL_V22_SARASHINA_URL",
        description="軽量な国産3Bモデル。Frozen v1実測 66/100。",
    ),
    SemanticModelSpec(
        key="llm-jp-4-8b",
        label="LLM-jp 4 8B Instruct",
        short_label="LLM-jp 8B",
        model_id="llm-jp/llm-jp-4-8b-instruct",
        endpoint_url="https://ryota-matsuki--ehime-kokubunsai-semantic-v2-2-api-llmjps-0cb1b6.modal.run",
        modal_url_secret="MODAL_V22_LLMJP_URL",
        description="NII LLM-jpの8B Instructモデル。Semantic Operations v2.2を共通利用。",
    ),
)

MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def get_model_spec(key: str | None) -> SemanticModelSpec:
    """Resolve a UI/backend model key with a stable default."""

    return MODEL_BY_KEY.get(str(key or ""), MODEL_BY_KEY[DEFAULT_MODEL_KEY])


def resolve_model_url(spec: SemanticModelSpec, secrets: Mapping[str, object]) -> str:
    """Resolve an optional deployment override, otherwise use the known URL."""

    value = secrets.get(spec.modal_url_secret)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return spec.endpoint_url


def configured_model_specs(secrets: Mapping[str, object]) -> tuple[SemanticModelSpec, ...]:
    """Return models with a known deployed or explicitly overridden endpoint."""

    return tuple(spec for spec in MODEL_SPECS if resolve_model_url(spec, secrets))


__all__ = [
    "DEFAULT_MODEL_KEY",
    "MODEL_BY_KEY",
    "MODEL_SPECS",
    "SemanticModelSpec",
    "configured_model_specs",
    "get_model_spec",
    "resolve_model_url",
]
