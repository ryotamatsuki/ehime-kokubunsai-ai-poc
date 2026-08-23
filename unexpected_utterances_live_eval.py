"""Manual Modal entrypoint for the frozen Unexpected User Utterances v1 baseline.

Do not add this command to automatic CI. The first run is intentionally manual
so the 100-case fixture remains unseen by production tuning until baseline data
has been captured.
"""

from __future__ import annotations

from typing import Any
import json

from modal_backend import app, EhimeCulturalGuide, MODEL_ID
from unexpected_utterances_eval import evaluate_dataset, load_dataset


@app.local_entrypoint()
def run(
    limit: int = 100,
    format_enforcer: str = "baseline",
    include_raw: bool = False,
    dataset_path: str = "tests/data/unexpected_utterances_v1/manifest.json",
):
    if format_enforcer not in {"baseline", "lmfe"}:
        raise ValueError("format_enforcer must be baseline or lmfe")
    dataset = load_dataset(dataset_path)
    guide = EhimeCulturalGuide()

    def invoke(payload: dict[str, Any]) -> Any:
        response = guide.generate_command.remote(
            payload.get("query", ""),
            payload.get("state"),
            payload.get("repair"),
            format_enforcer,
        )
        if not isinstance(response, dict):
            return None
        return response.get("answer")

    report = evaluate_dataset(
        dataset,
        invoke,
        limit=max(0, int(limit)),
        include_raw=include_raw,
        format_enforcer=format_enforcer,
    )
    report["model"] = MODEL_ID
    report["format_enforcer"] = format_enforcer
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)
    print(rendered)
    return rendered
