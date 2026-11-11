"""Cognitive Distillation Engine (`reflector.py`).

Sanitizes sensitive credentials/PII from stacktraces and payloads, synthesizes a
single-line deterministic procedural rule (<= 200 chars), and commits it to `AegisMemoryEngine`.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable

from aegis_mem.config import settings
from aegis_mem.core.engine import AegisMemoryEngine, get_engine
from aegis_mem.core.telemetry import telemetry
from aegis_mem.schemas import FailureReflectionInput, ReflectionResult

DISTILLATION_SYSTEM_PROMPT = (
    "You are the Cognitive Distillation Engine for aegis-mem. "
    "Analyze the provided execution failure traceback and input context. "
    "Output ONLY a single-line, deterministic, imperative procedural rule (maximum 200 characters) "
    "that tells the agent exactly what constraint, parameter format, or precondition to apply "
    "to prevent this failure in future executions. Do not include markdown, quotes, or explanations."
)

# Regex patterns to strip API keys, Bearer tokens, passwords, AWS keys, and emails
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer / Basic tokens
    (re.compile(r"(?i)(authorization\s*[:=]\s*['\"]?bearer\s+)[a-zA-Z0-9_\-\.~+/]+=*"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.~+/]{8,}=*"), r"\1[REDACTED_TOKEN]"),
    # OpenAI / Anthropic / Generic sk- keys
    (re.compile(r"\bsk-[a-zA-Z0-9_\-]{10,}\b"), "[REDACTED_API_KEY]"),
    # AWS Access Key IDs
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    # Key-value secret assignments (api_key, secret, password, token, passwd)
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|pwd|client[_-]?secret)"
            r"(\s*[:=]\s*['\"]?)([^'\"\s,&;}{]+)(['\"]?)"
        ),
        r"\1\2[REDACTED_SECRET]\4",
    ),
    # Connection strings with credentials (e.g., postgresql://user:pass@host)
    (re.compile(r"([a-zA-Z0-9+.-]+://[^:\s/]+:)([^@\s/]+)(@)"), r"\1[REDACTED_PASSWORD]\3"),
    # Email addresses (PII)
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
]


def sanitize_sensitive_text(raw_text: str) -> str:
    """Sanitize API keys, tokens, passwords, and PII from tracebacks and payloads."""
    sanitized = raw_text
    for pattern, replacement in _SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _normalize_single_line_rule(raw_rule: str, max_chars: int = 200) -> str:
    """Enforce a single-line deterministic rule of at most `max_chars` characters."""
    first_line = next((line.strip() for line in raw_rule.splitlines() if line.strip()), "")
    cleaned = first_line.strip(" \"'`*-•")
    if not cleaned:
        cleaned = "Validate input payload schema and endpoint preconditions before execution."
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _heuristic_distill(sanitized_error: str, sanitized_context: str) -> str:
    """Deterministic fallback synthesizer when an external LLM is not configured or mocked."""
    lower_err = sanitized_error.lower()
    if "422" in lower_err or "unprocessable" in lower_err or "validation" in lower_err:
        return _normalize_single_line_rule(
            f"Prevent HTTP 422 / validation error in ({sanitized_context[:60]}): enforce strict schema types and required fields ({sanitized_error[:70]})."
        )
    if "429" in lower_err or "rate limit" in lower_err or "too many requests" in lower_err:
        return _normalize_single_line_rule(
            f"Prevent HTTP 429 rate-limit on ({sanitized_context[:60]}): throttle requests and apply exponential backoff."
        )
    last_line = sanitized_error.strip().splitlines()[-1] if sanitized_error.strip() else "unknown error"
    return _normalize_single_line_rule(
        f"When executing '{sanitized_context[:55]}', avoid '{last_line[:85]}' by validating parameters beforehand."
    )


def _call_llm_for_distillation(
    sanitized_error: str,
    sanitized_context: str,
    engine: AegisMemoryEngine,
) -> str:
    """Invoke the underlying Mem0 LLM if available, falling back to deterministic synthesis."""
    user_prompt = (
        f"INPUT CONTEXT:\n{sanitized_context}\n\n"
        f"ERROR TRACE:\n{sanitized_error}\n\n"
        "Synthesize 1 imperative procedural rule (<= 200 chars):"
    )
    mem_obj = getattr(engine, "_memory", None)
    llm_client = getattr(mem_obj, "llm", None)
    if llm_client is not None and hasattr(llm_client, "generate_response"):
        try:
            response = llm_client.generate_response(
                messages=[
                    {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            if isinstance(response, str) and response.strip():
                return _normalize_single_line_rule(response)
        except Exception:
            pass
    return _heuristic_distill(sanitized_error, sanitized_context)


async def distill_failure(
    error_trace: str,
    input_context: str,
    agent_id: str,
    *,
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    engine: AegisMemoryEngine | None = None,
    llm_callable: Callable[[str, str], str] | None = None,
) -> ReflectionResult:
    """Sanitize secrets, distill a <=200-char procedural rule, and commit it to `AegisMemoryEngine`."""
    payload = FailureReflectionInput(
        agent_id=agent_id,
        error_trace=error_trace,
        input_context=input_context,
        tool_name=tool_name,
        metadata=metadata or {},
    )

    start = time.perf_counter()
    sanitized_err = sanitize_sensitive_text(payload.error_trace)
    sanitized_ctx = sanitize_sensitive_text(payload.input_context)

    active_engine = engine or get_engine()

    if llm_callable is not None:
        raw_rule = await asyncio.to_thread(llm_callable, sanitized_err, sanitized_ctx)
        distilled_rule = _normalize_single_line_rule(raw_rule)
    else:
        distilled_rule = await asyncio.to_thread(
            _call_llm_for_distillation, sanitized_err, sanitized_ctx, active_engine
        )

    rule_meta: dict[str, Any] = {
        "category": "error_remediation",
        "type": "procedural_rule",
        "source": "self_reflection",
        **payload.metadata,
    }
    if payload.tool_name:
        rule_meta["tool_name"] = payload.tool_name

    store_resp = await active_engine.store_rule(
        agent_id=payload.agent_id,
        rule_text=distilled_rule,
        metadata=rule_meta,
    )

    memory_id: str | None = None
    results_list = store_resp.get("results") if isinstance(store_resp, dict) else None
    if isinstance(results_list, list) and results_list:
        first_item = results_list[0]
        if isinstance(first_item, dict):
            memory_id = str(first_item.get("id", "")) or None
    elif isinstance(store_resp, dict) and "id" in store_resp:
        memory_id = str(store_resp["id"])

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    await telemetry.log_event(
        event_type="distill_failure",
        agent_id=payload.agent_id,
        latency_ms=elapsed_ms,
        success=True,
        details={
            "tool_name": payload.tool_name,
            "distilled_rule": distilled_rule,
            "memory_id": memory_id,
        },
    )

    return ReflectionResult(
        agent_id=payload.agent_id,
        sanitized_error=sanitized_err,
        distilled_rule=distilled_rule,
        memory_id=memory_id,
        stored=True,
    )

