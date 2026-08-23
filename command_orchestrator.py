"""Public Command orchestrator with deterministic suitability preflight.

The implementation lives in :mod:`command_orchestrator_core`.  This thin
facade keeps the existing public API stable while intercepting only ambiguous
suitability requests that the current event data cannot ground safely.
"""

from __future__ import annotations

import command_orchestrator_core as _core
import suitability_clarification as _suitability

# Re-export the existing public and compatibility surface, including the
# private helpers used by the repository's regression tests.  The core file is
# byte-for-byte the previous command_orchestrator.py implementation.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


class CommandOrchestrator(_core.CommandOrchestrator):
    """Add a bounded preflight before the existing semantic-command pipeline."""

    def handle_query(self, query, state=None, *, command_plan=None):
        if command_plan is None and isinstance(query, str):
            decision = _suitability.analyze_suitability_request(query)
            if decision.needs_clarification:
                plan = CommandPlan(
                    flow="unsupported",
                    slots=CommandSlots(),
                    confidence="high",
                )
                return CommandTurnResult(
                    status="clarification",
                    command=plan,
                    flow="unsupported",
                    message=_suitability.clarification_message(query),
                    observability={
                        "deterministic_route": "ambiguous_suitability_guard",
                        "deterministic_confidence": "high",
                    },
                    handled=True,
                )
            if decision.should_strip_suitability_marker:
                query = decision.sanitized_query
        return super().handle_query(query, state, command_plan=command_plan)


def handle_command_query(
    query,
    state=None,
    *,
    modal_call=None,
    reference_date=POC_REFERENCE_DATE,
    command_plan=None,
    events=None,
    output_format=DEFAULT_COMMAND_FORMAT,
    **kwargs,
):
    """Functional entrypoint preserving the previous call contract."""

    return CommandOrchestrator(
        modal_call,
        reference_date=reference_date,
        events=events,
        output_format=output_format,
    ).handle_query(query, state, command_plan=command_plan)


handle_command = handle_command_query
run_command = handle_command_query
execute_command = handle_command_query
orchestrate_command = handle_command_query
