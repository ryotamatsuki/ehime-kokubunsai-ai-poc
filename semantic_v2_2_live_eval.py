"""Isolated Sarashina + LMFE live evaluator for Semantic Operations v2.2.

Evaluation-only module.  It has no HTTP endpoint, does not modify production,
and exposes only the already-known frozen-v1 regression entrypoint.  The sealed
200-case holdout is intentionally unreachable from this module.

Uncertainty telemetry is observational only: token/atom margins never alter the
v2.2 decision path in this phase.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from typing import Any, Mapping, Sequence

import modal

from semantic_atomic_v2_2 import ATOMIC_FRAME_JSON_SCHEMA, EXPERIENCE_CONCEPTS
from semantic_prompt_v2_2 import build_atomic_frame_messages


MODEL_ID = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
MODEL_DIR = "/models/sarashina"
MODEL_VOLUME_NAME = "ehime-kokubunsai-model-cache"
MAX_INPUT_TOKENS = 7200
FRAME_MAX_NEW_TOKENS = 220
SERVICE_ID = "ehime-kokubunsai-semantic-v2-2-eval"
PROTOCOL_VERSION = "semantic-v2.2-atomic-fewshot-verifier-lmfe-preholdout-v1"
SEALED_HOLDOUT_SHA256 = "c844dda17248c0e7f16cd2985652e62bb0f8b601bf21196d6801479580899c92"

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
    "semantic_prompt_v2_2",
)

app = modal.App(SERVICE_ID)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.46.3,<5",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "lm-format-enforcer>=0.10,<1",
    )
    .add_local_python_source(*LOCAL_SOURCE_MODULES)
    .add_local_dir("data", remote_path="/root/data")
)


def _field_value_spans(answer: str) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    simple_fields = (
        "intent", "scope", "municipality", "region", "fee", "reservation",
        "venue", "rain", "audience_mode", "clarification", "data_gap",
    )
    for field in simple_fields:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', answer)
        if match is not None:
            spans[field] = match.span(1)
    for concept in EXPERIENCE_CONCEPTS:
        match = re.search(rf'"{re.escape(concept)}"\s*:\s*"([^"]*)"', answer)
        if match is not None:
            spans[f"experience.{concept}"] = match.span(1)
    return spans


def _token_char_spans(tokenizer: Any, token_ids: Sequence[int]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    previous = ""
    prefix: list[int] = []
    for token_id in token_ids:
        prefix.append(int(token_id))
        current = tokenizer.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        start = len(previous)
        end = len(current)
        spans.append((start, end, current[start:end]))
        previous = current
    return spans


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _summarize_uncertainty(
    tokenizer: Any,
    raw_answer: str,
    generated_ids: Sequence[int],
    scores: Sequence[Any],
    allowed_by_step: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Summarize LMFE-constrained token uncertainty without changing behavior."""

    import torch

    char_spans = _token_char_spans(tokenizer, generated_ids)
    field_spans = _field_value_spans(raw_answer)
    steps: list[dict[str, Any]] = []
    field_steps: dict[str, list[dict[str, float]]] = {field: [] for field in field_spans}

    for index, score_tensor in enumerate(scores):
        if index >= len(generated_ids):
            break
        logits = score_tensor[0].float()
        allowed = tuple(int(token_id) for token_id in allowed_by_step.get(index, ()))
        if allowed:
            allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.long)
            candidate_logits = logits.index_select(0, allowed_tensor)
            candidate_ids = allowed
        else:
            top_values, top_indices = torch.topk(logits, k=min(64, int(logits.shape[-1])))
            candidate_logits = top_values
            candidate_ids = tuple(int(item) for item in top_indices.tolist())

        finite = torch.isfinite(candidate_logits)
        if not bool(finite.any()):
            continue
        candidate_logits = candidate_logits[finite]
        candidate_ids = tuple(
            token_id for token_id, keep in zip(candidate_ids, finite.tolist()) if bool(keep)
        )
        probabilities = torch.softmax(candidate_logits, dim=-1)
        count = int(probabilities.numel())
        if count == 0:
            continue
        topk = torch.topk(probabilities, k=min(2, count)).values.tolist()
        top1 = float(topk[0])
        top2 = float(topk[1]) if len(topk) > 1 else 0.0
        margin = top1 - top2
        entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())
        selected_id = int(generated_ids[index])
        selected_probability = None
        if selected_id in candidate_ids:
            selected_probability = float(probabilities[candidate_ids.index(selected_id)].item())

        token_span = (char_spans[index][0], char_spans[index][1]) if index < len(char_spans) else (0, 0)
        field = next((name for name, span in field_spans.items() if _overlap(token_span, span)), None)
        row = {
            "step": index,
            "allowed_count": len(allowed) if allowed else count,
            "top1_probability": round(top1, 6),
            "top2_probability": round(top2, 6),
            "margin": round(margin, 6),
            "entropy": round(entropy, 6),
            "selected_probability": round(selected_probability, 6) if selected_probability is not None else None,
            "field": field,
        }
        steps.append(row)
        if field is not None and count > 1:
            field_steps[field].append({
                "margin": margin,
                "entropy": entropy,
                "selected_probability": selected_probability if selected_probability is not None else 0.0,
            })

    decision_steps = [row for row in steps if int(row["allowed_count"]) > 1]
    fields: dict[str, Any] = {}
    for field, values in field_steps.items():
        if not values:
            continue
        margins = [float(item["margin"]) for item in values]
        entropies = [float(item["entropy"]) for item in values]
        selected = [float(item["selected_probability"]) for item in values]
        fields[field] = {
            "decision_steps": len(values),
            "min_margin": round(min(margins), 6),
            "mean_margin": round(sum(margins) / len(margins), 6),
            "max_entropy": round(max(entropies), 6),
            "min_selected_probability": round(min(selected), 6),
        }

    margins = [float(row["margin"]) for row in decision_steps]
    entropies = [float(row["entropy"]) for row in decision_steps]
    return {
        "method": "lmfe_allowed_token_logits_v1",
        "behavioral_threshold_enabled": False,
        "generated_steps": len(steps),
        "decision_steps": len(decision_steps),
        "min_margin": round(min(margins), 6) if margins else None,
        "mean_margin": round(sum(margins) / len(margins), 6) if margins else None,
        "max_entropy": round(max(entropies), 6) if entropies else None,
        "fields": fields,
    }


@app.cls(
    image=inference_image,
    gpu="T4",
    volumes={"/models": model_volume},
    max_containers=1,
    scaledown_window=60,
    timeout=600,
)
class AtomicSemanticFrameGuideV22:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        started = time.perf_counter()
        tokenizer_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.tokenizer_load_ms = (time.perf_counter() - tokenizer_started) * 1000

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_started = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model_load_only_ms = (time.perf_counter() - model_started) * 1000

        lmfe_started = time.perf_counter()
        parser = JsonSchemaParser(ATOMIC_FRAME_JSON_SCHEMA)
        self.lmfe_prefix = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        self.lmfe_setup_ms = (time.perf_counter() - lmfe_started) * 1000
        self.container_setup_ms = (time.perf_counter() - started) * 1000

    def _generate(self, messages: list[dict[str, str]], format_enforcer: str) -> tuple[str, dict[str, Any]]:
        import torch

        encode_started = time.perf_counter()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        prompt_tokens = int(input_ids.shape[-1])
        if prompt_tokens > MAX_INPUT_TOKENS:
            raise ValueError("input_too_long")

        allowed_by_step: dict[int, tuple[int, ...]] = {}

        def observed_prefix(batch_id: int, current_ids: Any):
            allowed = self.lmfe_prefix(batch_id, current_ids)
            if batch_id == 0:
                step = max(0, int(current_ids.shape[-1]) - prompt_tokens)
                allowed_by_step[step] = tuple(int(token_id) for token_id in allowed)
            return allowed

        kwargs: dict[str, Any] = {
            "max_new_tokens": FRAME_MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": 1.02,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if format_enforcer == "lmfe":
            kwargs["prefix_allowed_tokens_fn"] = observed_prefix
        elif format_enforcer != "baseline":
            raise ValueError("unsupported_format_enforcer")

        generation_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(input_ids, **kwargs)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        generated_tensor = outputs.sequences[0, input_ids.shape[-1]:]
        generated_ids = [int(item) for item in generated_tensor.tolist()]
        decode_started = time.perf_counter()
        raw_answer = self.tokenizer.decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = raw_answer.strip()
        decode_ms = (time.perf_counter() - decode_started) * 1000

        uncertainty = _summarize_uncertainty(
            self.tokenizer,
            raw_answer,
            generated_ids,
            list(outputs.scores),
            allowed_by_step,
        )
        return answer, {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated_ids),
            "encode_ms": round(encode_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "uncertainty": uncertainty,
        }

    @modal.method()
    def generate_frame(
        self,
        query: str,
        state: Any = None,
        grounded: Any = None,
        format_enforcer: str = "lmfe",
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        value = str(query).strip()
        base_obs: dict[str, Any] = {
            "request_id": request_id,
            "service_id": SERVICE_ID,
            "protocol_version": PROTOCOL_VERSION,
            "model_id": MODEL_ID,
            "format_enforcer": format_enforcer,
            "repair": False,
            "container_setup_ms": round(float(self.container_setup_ms), 3),
            "tokenizer_load_ms": round(float(self.tokenizer_load_ms), 3),
            "model_load_ms": round(float(self.model_load_only_ms), 3),
            "lmfe_setup_ms": round(float(self.lmfe_setup_ms), 3),
        }
        if not value:
            base_obs["server_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return {"service_id": SERVICE_ID, "error": "質問がありません", "observability": base_obs}
        if format_enforcer not in {"baseline", "lmfe"}:
            base_obs["server_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return {"service_id": SERVICE_ID, "error": "形式制約モードが正しくありません。", "observability": base_obs}

        messages_started = time.perf_counter()
        messages = build_atomic_frame_messages(
            value,
            state if isinstance(state, Mapping) else {},
            grounded if isinstance(grounded, Mapping) else {},
        )
        base_obs["message_build_ms"] = round((time.perf_counter() - messages_started) * 1000, 3)
        try:
            answer, generation_obs = self._generate(messages, format_enforcer)
        except ValueError as exc:
            base_obs["server_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            base_obs["error_type"] = str(exc)
            return {"service_id": SERVICE_ID, "error": str(exc), "observability": base_obs}
        base_obs.update(generation_obs)
        base_obs["server_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return {"service_id": SERVICE_ID, "answer": answer, "observability": base_obs}


@app.local_entrypoint()
def frozen_v1_eval(
    limit: int = 100,
    format_enforcer: str = "lmfe",
    include_raw: bool = True,
):
    """Run only the exposed frozen-v1 regression; never the sealed holdout."""

    from unexpected_utterances_v2_2_eval import evaluate_frozen_v1_v22

    if format_enforcer not in {"lmfe", "baseline"}:
        raise ValueError("format_enforcer must be lmfe or baseline")
    guide = AtomicSemanticFrameGuideV22()

    def invoke(payload: Mapping[str, Any]) -> Any:
        return guide.generate_frame.remote(
            str(payload.get("query", "")),
            payload.get("state"),
            payload.get("grounded"),
            str(payload.get("format_enforcer", format_enforcer)),
        )

    result = evaluate_frozen_v1_v22(
        invoke,
        limit=max(0, min(100, int(limit))),
        include_raw=bool(include_raw),
        format_enforcer=format_enforcer,
    )
    result["run_metadata"] = {
        "protocol_version": PROTOCOL_VERSION,
        "service_id": SERVICE_ID,
        "model_id": MODEL_ID,
        "gpu": "T4",
        "max_new_tokens": FRAME_MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.02,
        "few_shot_examples": 5,
        "uncertainty_behavioral_threshold_enabled": False,
        "sealed_holdout_opened": False,
        "sealed_holdout_sha256": SEALED_HOLDOUT_SHA256,
    }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(rendered)
    return rendered
