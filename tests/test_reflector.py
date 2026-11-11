"""Tests for `reflector.py`, `@auto_learn` decorator, FastMCP tools, and FastAPI endpoints."""

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

