"""Tests for `reflector.py`, SHA-256 circuit breaker, `@auto_learn` decorator, FastMCP tools, and FastAPI endpoints."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from aegis_mem.core.engine import AegisMemoryEngine
from aegis_mem.core.reflector import distill_failure, sanitize_sensitive_text
from aegis_mem.interfaces.cloud_api import app
from aegis_mem.interfaces.mcp_server import (
    commit_learned_fact,
    get_procedural_memories,
    record_execution_failure,
    report_rule_outcome,
)
from aegis_mem.utils.decorators import auto_learn


def test_sanitize_sensitive_text_strips_credentials_and_pii() -> None:
    """Verify API keys, Bearer tokens, DB passwords, and emails are scrubbed before LLM call."""
    raw = (
        "HTTP 401 from https://api.example.com with Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 "
        "and api_key='sk-proj-1234567890abcdef123456' for user admin@corp.internal "
        "on postgresql://dbadmin:SuperSecret99!@db.internal:5432/prod"
    )
    cleaned = sanitize_sensitive_text(raw)

    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in cleaned
    assert "sk-proj-1234567890abcdef123456" not in cleaned
    assert "admin@corp.internal" not in cleaned
    assert "SuperSecret99!" not in cleaned
    assert "[REDACTED_TOKEN]" in cleaned
    assert "[REDACTED_SECRET]" in cleaned or "[REDACTED_API_KEY]" in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_PASSWORD]" in cleaned


@pytest.mark.asyncio
async def test_distill_failure_stores_recoverable_procedural_rule(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify a simulated HTTP 422 failure is sanitized, distilled into <=200 chars, and recalled."""
    error_trace = (
        "HTTPStatusError: 422 Unprocessable Entity on POST /v1/invoices "
        "(Authorization: Bearer secret_jwt_token_123456789): "
        '{"detail": [{"loc": ["body", "start_date"], "msg": "Invalid date format, expected YYYY-MM-DD"}]}'
    )
    input_context = 'POST /v1/invoices payload={"start_date": "11/10/2026", "user": "ops@company.com"}'

    reflection = await distill_failure(
        error_trace=error_trace,
        input_context=input_context,
        agent_id="invoice-agent",
        tool_name="create_invoice",
        engine=in_memory_engine,
    )

    assert reflection.stored is True
    assert reflection.deduplicated is False
    assert len(reflection.distilled_rule) <= 200
    assert "\n" not in reflection.distilled_rule
    assert "secret_jwt_token_123456789" not in reflection.sanitized_error
    assert "ops@company.com" not in reflection.distilled_rule

    # Now perform pre-flight recall for the same agent
    recalled = await in_memory_engine.recall_rules(
        agent_id="invoice-agent",
        task_query="POST /v1/invoices with start_date parameter",
        limit=3,
    )
    assert len(recalled) == 1
    assert "ISO-8601" in recalled[0].rule or "422" in recalled[0].rule


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_duplicate_error_burst(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify 10 identical exceptions (with volatile request IDs/timestamps) trigger only 1 LLM distillation."""
    llm_call_count = 0

    def tracked_llm(sanitized_err: str, sanitized_ctx: str) -> str:
        nonlocal llm_call_count
        llm_call_count += 1
        return "Route /v1/orders to standby gateway when HTTP 502 Bad Gateway occurs."

    results = []
    for i in range(10):
        # Volatile request IDs and timestamps change each iteration, but normalized fingerprint is identical
        trace = (
            f"HTTPStatusError: HTTP 502 Bad Gateway on POST /v1/orders "
            f"[req_id=550e8400-e29b-41d4-a716-44665544000{i} ts=2026-11-11T00:10:{10 + i:02d}Z retry={100 + i}]"
        )
        res = await distill_failure(
            error_trace=trace,
            input_context="POST /v1/orders",
            agent_id="burst-agent",
            tool_name="submit_order",
            engine=in_memory_engine,
            llm_callable=tracked_llm,
            cooldown_seconds=300.0,
        )
        results.append(res)

    assert llm_call_count == 1
    assert results[0].stored is True
    assert results[0].deduplicated is False
    assert all(r.deduplicated is True and r.stored is False for r in results[1:])

    recalled = await in_memory_engine.recall_rules(
        agent_id="burst-agent",
        task_query="POST /v1/orders gateway error",
    )
    assert len(recalled) == 1


@pytest.mark.asyncio
async def test_auto_learn_decorator_captures_exception_and_commits_rule(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify `@auto_learn` triggers background reflection on exception and makes the rule recallable."""

    @auto_learn(agent_id="decorator-agent", reraise=True, engine=in_memory_engine)
    async def fetch_invoices(start_date: str, api_key: str) -> dict[str, str]:
        raise RuntimeError(
            f"HTTP 422 Unprocessable Entity: start_date={start_date} invalid for api_key={api_key}"
        )

    with pytest.raises(RuntimeError, match="HTTP 422"):
        await fetch_invoices("10-11-2026", api_key="sk-live-9999999999999999")

    # Wait briefly for the background asyncio task created by @auto_learn to finish
    await asyncio.sleep(0.05)

    recalled = await in_memory_engine.recall_rules(
        agent_id="decorator-agent",
        task_query="fetch_invoices start_date format",
    )
    assert len(recalled) == 1
    assert len(recalled[0].rule) <= 200


@pytest.mark.asyncio
async def test_mcp_tools_and_fastapi_endpoints(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify FastMCP tools and FastAPI endpoints adhere to ToolResponse contracts."""
    mcp_fn_record = getattr(record_execution_failure, "fn", record_execution_failure)
    mcp_fn_recall = getattr(get_procedural_memories, "fn", get_procedural_memories)
    mcp_fn_commit = getattr(commit_learned_fact, "fn", commit_learned_fact)
    mcp_fn_outcome = getattr(report_rule_outcome, "fn", report_rule_outcome)

    rec_res = await mcp_fn_record(
        agent_id="mcp-agent",
        error_trace="HTTP 429 Too Many Requests on /v1/search",
        input_context="GET /v1/search?q=logs",
        tool_name="search_logs",
    )
    assert rec_res["success"] is True

    commit_res = await mcp_fn_commit(
        agent_id="mcp-agent",
        rule_text="Include X-Tenant-ID header on all /v1/search calls.",
        category="schema_constraint",
    )
    assert commit_res["success"] is True

    recall_res = await mcp_fn_recall(
        agent_id="mcp-agent",
        task_description="Call /v1/search endpoint",
        top_k=5,
    )
    assert recall_res["success"] is True
    assert recall_res["data"]["count"] == 2

    first_mem_id = recall_res["data"]["rules"][0]["memory_id"]
    outcome_res = await mcp_fn_outcome(
        agent_id="mcp-agent",
        memory_id=first_mem_id,
        succeeded=True,
    )
    assert outcome_res["success"] is True
    assert outcome_res["data"]["utility_score"] == pytest.approx(1.1)

    # FastAPI endpoints test
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    reflect_resp = client.post(
        "/v1/memories/reflect",
        json={
            "agent_id": "cloud-agent",
            "error_trace": "HTTP 422 on /v1/invoices date field",
            "input_context": "POST /v1/invoices",
            "tool_name": "post_invoice",
        },
    )
    assert reflect_resp.status_code == 202
    assert reflect_resp.json()["success"] is True

    recall_api_resp = client.post(
        "/v1/memories/recall",
        json={
            "agent_id": "cloud-agent",
            "task_description": "POST /v1/invoices",
            "top_k": 5,
        },
    )
    assert recall_api_resp.status_code == 200
    assert recall_api_resp.json()["data"]["count"] == 1

    cloud_mem_id = recall_api_resp.json()["data"]["rules"][0]["memory_id"]
    feedback_resp = client.post(
        "/v1/memories/feedback",
        json={
            "agent_id": "cloud-agent",
            "memory_id": cloud_mem_id,
            "succeeded": True,
        },
    )
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["data"]["utility_score"] == pytest.approx(1.1)
