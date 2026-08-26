"""Microbenchmark for deterministic Evidence-Bounded v2.3 layers.

This intentionally excludes model/network/executor latency. It measures only
Capability Registry lookup and the pure-Python evidence verifier after import
warm-up so the architecture can demonstrate that the new deterministic guard
is negligible relative to one LLM generation.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Callable

from semantic_atomic_v2_2 import neutral_experience
from semantic_atomic_v2_3 import AtomicSemanticFrameV23
from semantic_capability_registry_v2_3 import lookup_capability
from semantic_evidence_v2_3 import EvidenceRequest, SemanticResolution
from semantic_verifier_v2_3 import verify_evidence_bounded_frame


def _summary_ms(samples_ns: list[int]) -> dict[str, float]:
    values = sorted(value / 1_000_000 for value in samples_ns)
    if not values:
        return {"median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * 0.95))))
    return {
        "median_ms": round(float(statistics.median(values)), 6),
        "p95_ms": round(float(values[index]), 6),
        "max_ms": round(float(values[-1]), 6),
    }


def _measure(call: Callable[[], object], iterations: int) -> dict[str, float]:
    for _ in range(200):
        call()
    samples: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - started)
    return _summary_ms(samples)


def benchmark_deterministic_layers(iterations: int = 3000) -> dict[str, object]:
    count = max(200, int(iterations))
    frame = AtomicSemanticFrameV23(
        intent="search",
        scope="new",
        evidence_request=EvidenceRequest.SUPPORTED_ATTRIBUTE.value,
        semantic_resolution=SemanticResolution.RESOLVED.value,
        municipality="none",
        region="none",
        fee="none",
        reservation="none",
        venue="none",
        rain="none",
        audience_mode="none",
        experience={**neutral_experience(), "watch_listen": "require"},
    )
    query = "展示を見て解説を聞くことを中心に楽しめる催し"
    grounded = {"experience_required": ["watch_listen"]}

    registry = _measure(lambda: lookup_capability(EvidenceRequest.RELATIONAL_SUITABILITY), count)
    verifier = _measure(
        lambda: verify_evidence_bounded_frame(frame, query=query, state=None, grounded=grounded),
        count,
    )
    return {
        "iterations": count,
        "scope": "pure_python_registry_and_verifier_only",
        "capability_registry": registry,
        "evidence_verifier": verifier,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark_deterministic_layers(), ensure_ascii=False, sort_keys=True))


__all__ = ["benchmark_deterministic_layers"]
