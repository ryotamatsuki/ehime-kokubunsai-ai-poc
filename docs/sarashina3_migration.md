# Sarashina3 mini API migration — Stage 1

Status: **opt-in transport preparation only; no live provider cutover**.

## Scope

- The existing Streamlit application, Modal endpoint, model image, search engine,
  semantic-command validator, event facts and deployment workflow are unchanged.
- `llm_client.py` adds:
  - `ModalTransport`: preserves the current authenticated PoC Modal wire contract.
  - `SarashinaChatCompletionsTransport`: an **assumption-dependent**
    OpenAI-style chat-completions adapter, not yet connected to Streamlit.
  - `GenerationRequest`: one call contract carrying both an existing Modal
    payload and provider-neutral `system/user/assistant` messages.
  - Credential-free, offline tests (`tests/test_llm_client.py`).
- Neither transport sends HTTP until its `generate()` method is explicitly invoked.
- **No production activation** is possible in Stage 1 because the Streamlit
  entry point still calls its existing Modal implementation.

## Critical verification gate

SB Intuitions announced the Sarashina3 series on Cloud PF Type A in June 2026.
The public announcement does **not** establish the exact endpoint, model ID,
authentication header, rate limit, JSON mode or chat response schema for the
**13 October 2026 free beta**.

Before any live request or Stage-2 cutover:

1. Obtain the API-specific technical documentation from the beta console.
2. Verify the required URL, hostname, model identifier and authentication
   header format.
3. Confirm whether Chat Completions is OpenAI-compatible, including
   request parameters (`max_tokens`, `temperature`) and
   `choices[0].message.content` responses.
4. Confirm JSON schema or response-format controls, context limits,
   request/response retention, permitted input content and usage quotas.
5. If any aspect differs, update the isolated transport and its mock tests
   **before connecting it to the application**. Do not guess a URL or model ID.

The transport expects all of the following explicit values, and **none**
are stored in Git:

- `SARASHINA_CHAT_COMPLETIONS_URL`: full provider-confirmed HTTPS URL
- `SARASHINA_API_HOST`: exact allow-listed hostname (must match URL)
- `SARASHINA_MODEL_ID`: provider-confirmed exact model ID
- `SARASHINA_API_KEY`: credential from the beta console
- `SARASHINA_AUTH_HEADER`: provider-confirmed `Authorization`,
  `x-api-key`, or `api-key`
- `SARASHINA_AUTH_SCHEME`: provider-confirmed `Bearer` or `Raw`

These are **planned configuration names**, not active Streamlit secrets yet.
The URL/model/header in the unit tests are deliberately fictional fixtures.
Currently they do not assert the actual Sarashina API interface.

## Planned Stage-2 wiring

1. Extract system prompts and input sanitizers from `modal_backend.py`
   into provider-neutral, unit-testable message builders. Preserve
   `COMMAND_SYSTEM_PROMPT`, Planner/Writer constraints and bounded
   event context.
2. Adapt `agent_planner.py`, `streamlit_app.py` and the command callback
   to call a common transport. Default to `modal`, with a separate explicit
   opt-in `sarashina_api` switch; fail closed if API credentials are missing.
3. Keep `command_generator.py` parse/repair (max one repair), deterministic
   `event_search.py`, bounded tool dispatcher, response factual guards and
   event IDs fully local.
4. Update observability to use provider-neutral failure categories without
   storing prompts, API keys, response bodies or personal information.
5. Extend security tests to assert provider-specific authentication, approved
   host/HTTPS enforcement and no URL injection. Never disable the old
   Modal security tests in a way that weakens production coverage.
6. Extend the existing *manual* live evaluation workflow to score both providers
   on the same held-out questions, including first-pass JSON validity,
   flow accuracy, fallback, median/p95 latency and multistep dialogues.
7. Keep the current Modal path as a tested rollback mechanism until an explicit
   review approves cutover.

## Local checks

```bash
python -m pip install 'requests>=2.32,<3' pytest
python -m pytest -q tests/test_llm_client.py
python -m compileall -q llm_client.py
```

GitHub's existing deterministic QA workflow also runs the new tests via
`python -m pytest -q`. These tests use injected fake HTTP functions; **no**
real Sarashina API request, API key, billing or provider comparison has
occurred during Stage 1.
