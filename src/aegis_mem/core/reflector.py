"""Cognitive Distillation Engine (`reflector.py`).

Sanitizes sensitive credentials/PII from stacktraces and payloads, enforces a pre-LLM
SHA-256 deduplication circuit breaker with TTL cooldown, synthesizes a single-line
deterministic procedural rule (<= 200 chars), and commits it to `AegisMemoryEngine`.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, Callable

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

# Normalization patterns for deterministic SHA-256 fingerprinting (UUIDs, ISO dates, hex addresses, volatile numbers)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
# Preserve common HTTP status codes (400-599) while normalizing arbitrary volatile numbers (req_ids, epochs, line numbers)
_VOLATILE_NUM_RE = re.compile(r"\b(?!4\d{2}\b|5\d{2}\b)\d{2,}\b")

# Pre-LLM SHA-256 Deduplication Circuit Breaker state
_dedup_cache: dict[str, float] = {}
_dedup_results: dict[str, ReflectionResult] = {}


def clear_dedup_cache() -> None:
    """Reset the in-memory circuit breaker deduplication cache (used in test suites)."""
    _dedup_cache.clear()
    _dedup_results.clear()


def sanitize_sensitive_text(raw_text: str) -> str:
    """Sanitize API keys, tokens, passwords, and PII from tracebacks and payloads."""
    sanitized = raw_text
    for pattern, replacement in _SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _extract_exception_type(sanitized_error: str) -> str:
    """Extract the leading or trailing exception/status class from a sanitized traceback."""
    lines = [ln.strip() for ln in sanitized_error.splitlines() if ln.strip()]
    if not lines:
        return "UnknownError"
    last_line = lines[-1]
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Fault|Status))", last_line)
    if match:
        return match.group(1)
    http_match = re.search(r"\bHTTP\s*([45]\d{2})\b", sanitized_error, re.IGNORECASE)
    if http_match:
        return f"HTTP_{http_match.group(1)}"
    first_token = last_line.split(":", 1)[0].strip()
    if first_token and len(first_token) <= 48 and " " not in first_token:
        return first_token
    return "ExecutionError"


def compute_error_fingerprint(agent_id: str, sanitized_error: str) -> str:
    """Compute `sha256(f'{agent_id}:{exception_type}:{normalized_sanitized_error}')`."""
    exc_type = _extract_exception_type(sanitized_error)
    normalized = _UUID_RE.sub("<UUID>", sanitized_error)
    normalized = _ISO_TS_RE.sub("<TS>", normalized)
    normalized = _HEX_RE.sub("<HEX>", normalized)
    normalized = _VOLATILE_NUM_RE.sub("<NUM>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    raw_key = f"{agent_id.strip()}:{exc_type}:{normalized}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


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
    if "502" in lower_err or "503" in lower_err or "bad gateway" in lower_err:
        return _normalize_single_line_rule(
            f"Mitigate HTTP 502/503 gateway fault on ({sanitized_context[:60]}): switch to fallback replica or circuit-break retries."
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
    cooldown_seconds: float = 300.0,
) -> ReflectionResult:
    """Sanitize secrets, check SHA-256 circuit breaker, distill <=200-char rule, and commit to `AegisMemoryEngine`."""
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

    # Pre-LLM SHA-256 Deduplication Circuit Breaker check
    fingerprint = compute_error_fingerprint(payload.agent_id, sanitized_err)
    now_mono = time.monotonic()
    last_seen = _dedup_cache.get(fingerprint)
    if (
        cooldown_seconds > 0
        and last_seen is not None
        and (now_mono - last_seen) < cooldown_seconds
        and fingerprint in _dedup_results
    ):
        cached = _dedup_results[fingerprint]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        await telemetry.log_event(
            event_type="circuit_breaker_dedup_hit",
            agent_id=payload.agent_id,
            latency_ms=elapsed_ms,
            success=True,
            details={
                "fingerprint": fingerprint,
                "tool_name": payload.tool_name,
                "llm_tokens_consumed": 0,
                "cached_memory_id": cached.memory_id,
            },
        )
        return ReflectionResult(
            agent_id=payload.agent_id,
            sanitized_error=sanitized_err,
            distilled_rule=cached.distilled_rule,
            memory_id=cached.memory_id,
            stored=False,
            deduplicated=True,
            fingerprint=fingerprint,
        )

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
        "fingerprint": fingerprint,
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
            "fingerprint": fingerprint,
        },
    )

    result = ReflectionResult(
        agent_id=payload.agent_id,
        sanitized_error=sanitized_err,
        distilled_rule=distilled_rule,
        memory_id=memory_id,
        stored=True,
        deduplicated=False,
        fingerprint=fingerprint,
    )
    _dedup_cache[fingerprint] = now_mono
    _dedup_results[fingerprint] = result
    return result
