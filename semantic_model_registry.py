"""Model registry for the Semantic Operations v2.2 PoC.

The semantic architecture, prompt, schema, verifier, reducer and executor are
shared across models.  Only the model backend changes.  Keeping that boundary
explicit makes the Streamlit selector a genuine model A/B control rather than
an architecture switch.
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
    modal_url_secret: str
    description: str


MODEL_SPECS: tuple[SemanticModelSpec, ...] = (
    SemanticModelSpec(
        key="sarashina-2.2-3b",
        label="Sarashina 2.2 3B",
        short_label="Sarashina 3B",
        model_id="sbintuitions/sarashina2.2-3b-instruct-v0.1",
        modal_url_secret="MODAL_V22_SARASHINA_URL",
        description="軽量な国産3Bモデル。Frozen v1実測 66/100。",
    ),
    SemanticModelSpec(
        key="llm-jp-4-8b",
        label="LLM-jp 4 8B Instruct",
        short_label="LLM-jp 8B",
        model_id="llm-jp/llm-jp-4-8b-instruct",
        modal_url_secret="MODAL_V22_LLMJP_URL",
        description="NII LLM-jpの8B Instructモデル。Semantic Operations v2.2を共通利用。",
    ),
)

MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def get_model_spec(key: str | None) -> SemanticModelSpec:
    """Resolve a UI/backend model key with a stable default."""

    return MODEL_BY_KEY.get(str(key or ""), MODEL_BY_KEY[DEFAULT_MODEL_KEY])


def configured_model_specs(secrets: Mapping[str, object]) -> tuple[SemanticModelSpec, ...]:
    """Return only models whose authenticated Semantic v2.2 endpoint is set."""

    configured: list[SemanticModelSpec] = []
    for spec in MODEL_SPECS:
        value = secrets.get(spec.modal_url_secret)
        if isinstance(value, str) and value.strip():
            configured.append(spec)
    return tuple(configured)


__all__ = [
    "DEFAULT_MODEL_KEY",
    "MODEL_BY_KEY",
    "MODEL_SPECS",
    "SemanticModelSpec",
    "configured_model_specs",
    "get_model_spec",
]
