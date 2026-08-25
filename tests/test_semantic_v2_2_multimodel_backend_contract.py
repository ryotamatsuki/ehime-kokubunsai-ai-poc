from __future__ import annotations

from pathlib import Path


def test_multimodel_backend_keeps_one_atomic_contract_for_both_models():
    source = Path("semantic_v2_2_multimodel_backend.py").read_text(encoding="utf-8")
    assert '"sbintuitions/sarashina2.2-3b-instruct-v0.1"' not in source
    assert '"llm-jp/llm-jp-4-8b-instruct"' not in source
    assert "MODEL_BY_KEY" in source
    assert "build_atomic_frame_messages" in source
    assert "ATOMIC_FRAME_JSON_SCHEMA" in source
    assert 'format_enforcer != "lmfe"' in source
    assert 'do_sample=False' in source
    assert 'repetition_penalty=1.02' in source
    assert 'max_new_tokens=FRAME_MAX_NEW_TOKENS' in source


def test_each_model_has_an_independent_t4_endpoint_and_volume():
    source = Path("semantic_v2_2_multimodel_backend.py").read_text(encoding="utf-8")
    assert "class SarashinaSemanticV22" in source
    assert "class LlmJpSemanticV22" in source
    assert source.count('gpu="T4"') == 2
    assert source.count('@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)') == 2
    assert source.count("async def semantic(self, body: dict[str, Any]):") == 2
    assert "await request.json()" not in source
    assert "from fastapi import Request" not in source
    assert "SARASHINA_VOLUME_NAME = \"ehime-kokubunsai-model-cache\"" in source
    assert "LLMJP_VOLUME_NAME = \"ehime-kokubunsai-llmjp-4-8b-cache\"" in source
    assert 'volumes={"/models": sarashina_volume}' in source
    assert 'volumes={"/models": llmjp_volume}' in source
    assert "download_llmjp" in source
    assert "download_sarashina" not in source


def test_backend_never_references_sealed_holdout():
    source = Path("semantic_v2_2_multimodel_backend.py").read_text(encoding="utf-8")
    assert "unexpected_utterances_holdout" not in source
    assert "holdout.json.gz" not in source
