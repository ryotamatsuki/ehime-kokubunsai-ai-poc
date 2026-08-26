"""Selectable Modal backends for Semantic Operations v2.3.

This module defines the next frozen backend contract but is not deployed by the
v2.3 implementation PR.  Both models share the same v2.3 schema, prompt, LMFE
constraint and decoding policy; only model key/cache/weights differ.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

import modal

from semantic_model_registry import MODEL_BY_KEY


SERVICE_ID = "ehime-kokubunsai-semantic-v2-3-api"
PROTOCOL_VERSION = "semantic-v2.3-evidence-bounded-lmfe-v1"
SARASHINA_VOLUME_NAME = "ehime-kokubunsai-model-cache"
LLMJP_VOLUME_NAME = "ehime-kokubunsai-llmjp-4-8b-cache"
SARASHINA_MODEL_KEY = "sarashina-2.2-3b"
LLMJP_MODEL_KEY = "llm-jp-4-8b"
SARASHINA_MODEL_DIR = "/models/sarashina"
LLMJP_MODEL_DIR = "/models/llm-jp-4-8b-instruct"
MAX_INPUT_TOKENS = 7200
FRAME_MAX_NEW_TOKENS = 220

LOCAL_SOURCE_MODULES = (
    "age_semantics",
    "app_config",
    "command_models",
    "data_model_v3",
    "event_details",
    "experience_matcher",
    "experience_preferences",
    "flow_registry",
    "semantic_atomic_v2_2",
    "semantic_atomic_v2_3",
    "semantic_evidence_v2_3",
    "semantic_model_registry",
    "semantic_prompt_v2_2",
    "semantic_prompt_v2_3",
)

app = modal.App(SERVICE_ID)
sarashina_volume = modal.Volume.from_name(SARASHINA_VOLUME_NAME, create_if_missing=False)
llmjp_volume = modal.Volume.from_name(LLMJP_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.46.3,<5",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "lm-format-enforcer>=0.10,<1",
        "fastapi[standard]",
    )
    .add_local_python_source(*LOCAL_SOURCE_MODULES)
    .add_local_dir("data", remote_path="/root/data")
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("semantic_model_registry")
)


@app.function(image=download_image, volumes={"/models": llmjp_volume}, timeout=7200)
def download_llmjp() -> None:
    from huggingface_hub import snapshot_download

    spec = MODEL_BY_KEY[LLMJP_MODEL_KEY]
    snapshot_download(
        repo_id=spec.model_id,
        local_dir=LLMJP_MODEL_DIR,
        allow_patterns=("*.json", "*.jinja", "*.py", "*.safetensors"),
    )
    llmjp_volume.commit()


def _load_runtime(model_dir: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    from semantic_atomic_v2_3 import ATOMIC_FRAME_JSON_SCHEMA_V23

    started = time.perf_counter()
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer_load_ms = (time.perf_counter() - tokenizer_started) * 1000
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_load_ms = (time.perf_counter() - model_started) * 1000
    lmfe_started = time.perf_counter()
    parser = JsonSchemaParser(ATOMIC_FRAME_JSON_SCHEMA_V23)
    lmfe_prefix = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "lmfe_prefix": lmfe_prefix,
        "tokenizer_load_ms": tokenizer_load_ms,
        "model_load_ms": model_load_ms,
        "lmfe_setup_ms": (time.perf_counter() - lmfe_started) * 1000,
        "container_setup_ms": (time.perf_counter() - started) * 1000,
    }


def _generate_frame(
    runtime: Mapping[str, Any],
    *,
    model_key: str,
    query: str,
    state: Any = None,
    grounded: Any = None,
    format_enforcer: str = "lmfe",
) -> dict[str, Any]:
    import torch
    from semantic_prompt_v2_3 import build_atomic_frame_messages_v23

    spec = MODEL_BY_KEY[model_key]
    started = time.perf_counter()
    observability: dict[str, Any] = {
        "request_id": uuid.uuid4().hex,
        "service_id": SERVICE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "model_id": spec.model_id,
        "format_enforcer": format_enforcer,
        "repair": False,
        "container_setup_ms": round(float(runtime["container_setup_ms"]), 3),
        "tokenizer_load_ms": round(float(runtime["tokenizer_load_ms"]), 3),
        "model_load_ms": round(float(runtime["model_load_ms"]), 3),
        "lmfe_setup_ms": round(float(runtime["lmfe_setup_ms"]), 3),
    }
    value = str(query).strip()
    if not value:
        return {"service_id": SERVICE_ID, "model_key": model_key, "model_id": spec.model_id, "error": "質問がありません", "observability": observability}
    if format_enforcer != "lmfe":
        return {"service_id": SERVICE_ID, "model_key": model_key, "model_id": spec.model_id, "error": "Semantic v2.3ではLMFEのみ利用できます。", "observability": observability}

    messages_started = time.perf_counter()
    messages = build_atomic_frame_messages_v23(
        value,
        state if isinstance(state, Mapping) else {},
        grounded if isinstance(grounded, Mapping) else {},
    )
    observability["message_build_ms"] = round((time.perf_counter() - messages_started) * 1000, 3)
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    encode_started = time.perf_counter()
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    observability["encode_ms"] = round((time.perf_counter() - encode_started) * 1000, 3)
    prompt_tokens = int(input_ids.shape[-1])
    observability["prompt_tokens"] = prompt_tokens
    if prompt_tokens > MAX_INPUT_TOKENS:
        return {"service_id": SERVICE_ID, "model_key": model_key, "model_id": spec.model_id, "error": "input_too_long", "observability": observability}

    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=FRAME_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=runtime["lmfe_prefix"],
        )
    observability["generation_ms"] = round((time.perf_counter() - generation_started) * 1000, 3)
    generated = output_ids[0, input_ids.shape[-1]:]
    observability["generated_tokens"] = int(generated.shape[-1])
    answer = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    observability["server_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return {
        "service_id": SERVICE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "model_id": spec.model_id,
        "answer": answer,
        "observability": observability,
    }


class _SemanticEndpointMixin:
    model_key: str
    model_dir: str

    def _enter(self) -> None:
        self.runtime = _load_runtime(self.model_dir)

    def _run(self, query: str, state: Any, grounded: Any, format_enforcer: str) -> dict[str, Any]:
        return _generate_frame(
            self.runtime,
            model_key=self.model_key,
            query=query,
            state=state,
            grounded=grounded,
            format_enforcer=format_enforcer,
        )


@app.cls(image=base_image, gpu="T4", volumes={"/models": sarashina_volume}, max_containers=1, scaledown_window=60, timeout=600)
class SarashinaSemanticV23(_SemanticEndpointMixin):
    model_key = SARASHINA_MODEL_KEY
    model_dir = SARASHINA_MODEL_DIR

    @modal.enter()
    def load_model(self) -> None:
        self._enter()

    @modal.method()
    def generate_frame(self, query: str, state: Any = None, grounded: Any = None, format_enforcer: str = "lmfe") -> dict[str, Any]:
        return self._run(query, state, grounded, format_enforcer)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def semantic(self, body: dict[str, Any]):
        return self._run(str(body.get("query", "")), body.get("state"), body.get("grounded"), str(body.get("format_enforcer", "lmfe")).lower())


@app.cls(image=base_image, gpu="T4", volumes={"/models": llmjp_volume}, max_containers=1, scaledown_window=60, timeout=600)
class LlmJpSemanticV23(_SemanticEndpointMixin):
    model_key = LLMJP_MODEL_KEY
    model_dir = LLMJP_MODEL_DIR

    @modal.enter()
    def load_model(self) -> None:
        self._enter()

    @modal.method()
    def generate_frame(self, query: str, state: Any = None, grounded: Any = None, format_enforcer: str = "lmfe") -> dict[str, Any]:
        return self._run(query, state, grounded, format_enforcer)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def semantic(self, body: dict[str, Any]):
        return self._run(str(body.get("query", "")), body.get("state"), body.get("grounded"), str(body.get("format_enforcer", "lmfe")).lower())


__all__ = [
    "FRAME_MAX_NEW_TOKENS",
    "LLMJP_MODEL_KEY",
    "LlmJpSemanticV23",
    "PROTOCOL_VERSION",
    "SARASHINA_MODEL_KEY",
    "SERVICE_ID",
    "SarashinaSemanticV23",
    "download_llmjp",
]
