"""FastMCP server exposing procedural memory recall, failure distillation, rule persistence, and utility feedback.

Follows strict layer separation: calls `core/engine.py` and `core/reflector.py` exclusively,
catches all exceptions, logs to `core/telemetry.py`, and returns `ToolResponse`.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from aegis_mem.core.engine import get_engine
from aegis_mem.core.reflector import distill_failure
from aegis_mem.core.telemetry import telemetry
from aegis_mem.schemas import MemoryCreate, MemorySearchQuery, RuleFeedbackInput, ToolResponse

mcp = FastMCP("aegis-mem")


@mcp.tool()
async def get_procedural_memories(
    agent_id: str,
    task_description: str,
    top_k: int = 5,
    user_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Pre-flight recall: Retrieve deterministic procedural rules learned from past executions.

    Call this tool BEFORE executing external API requests, SQL queries, or complex tool actions
    so the agent applies previously learned schema constraints and remediation rules.

    Args:
        agent_id: Unique identifier of the calling agent (mandatory scope).
        task_description: Natural-language description of the action or endpoint about to be called.
        top_k: Maximum number of procedural rules to return (default 5).
        user_id: Optional user scope.
        run_id: Optional execution run scope.
    """
    try:
        query = MemorySearchQuery(
            agent_id=agent_id,
            user_id=user_id,
            run_id=run_id,
            task_description=task_description,
            top_k=top_k,
        )
        rules = await get_engine().recall_context(query)
        return ToolResponse(
            success=True,
            data={"rules": [r.model_dump() for r in rules], "count": len(rules)},
        ).model_dump()
    except Exception as exc:
        await telemetry.log_event(
            event_type="mcp_get_procedural_memories_error",
            agent_id=agent_id or "unscoped",
            success=False,
            details={"error": str(exc)},
        )
        return ToolResponse(
            success=False,
            error_code="MEMORY_RECALL_FAILED",
            remediation_hint=(
                f"Ensure agent_id and task_description are non-empty strings. Details: {exc}"
            ),
        ).model_dump()


@mcp.tool()
async def record_execution_failure(
    agent_id: str,
    error_trace: str,
    input_context: str,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Self-reflection trigger: Distill an execution failure or stacktrace into a 1-line procedural rule.

    Call this tool whenever an external API call, database query, or tool invocation fails
    (e.g., HTTP 422 Unprocessable Entity, schema mismatch, rate limit). Secrets and PII are
    automatically stripped before distilling and committing the preventative rule.

    Args:
        agent_id: Unique identifier of the agent that experienced the failure.
        error_trace: Raw exception string, traceback, or HTTP error response body.
        input_context: The payload, arguments, or intent that triggered the error.
        tool_name: Optional identifier of the failed tool or endpoint.
    """
    try:
        reflection = await distill_failure(
            error_trace=error_trace,
            input_context=input_context,
            agent_id=agent_id,
            tool_name=tool_name,
        )
        return ToolResponse(
            success=True,
            data=reflection.model_dump(mode="json"),
        ).model_dump()
    except Exception as exc:
        await telemetry.log_event(
            event_type="mcp_record_execution_failure_error",
            agent_id=agent_id or "unscoped",
            success=False,
            details={"error": str(exc)},
        )
        return ToolResponse(
            success=False,
            error_code="FAILURE_DISTILLATION_ERROR",
            remediation_hint=(
                f"Verify agent_id, error_trace, and input_context are non-empty. Details: {exc}"
            ),
        ).model_dump()


@mcp.tool()
async def commit_learned_fact(
    agent_id: str,
    rule_text: str,
    category: str = "error_remediation",
    user_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Explicitly persist a verified procedural rule or operational constraint for an agent.

    Use this tool when the agent discovers an explicit domain rule or API constraint that should
    be remembered across future sessions without waiting for an exception.

    Args:
        agent_id: Unique identifier of the agent owning this procedural rule.
        rule_text: Concise, imperative rule statement (3 to 500 characters).
        category: Rule classification (e.g., 'error_remediation', 'schema_constraint', 'rate_limit').
        user_id: Optional user scope.
        run_id: Optional execution run scope.
    """
    try:
        payload = MemoryCreate(
            agent_id=agent_id,
            user_id=user_id,
            run_id=run_id,
            rule_text=rule_text,
            category=category,
        )
        stored = await get_engine().store_rule(
            agent_id=payload.agent_id,
            rule_text=payload.rule_text,
            metadata={"category": payload.category, "type": "procedural_rule"},
            user_id=payload.user_id,
            run_id=payload.run_id,
        )
        return ToolResponse(
            success=True,
            data={"stored": stored, "agent_id": payload.agent_id, "rule_text": payload.rule_text},
        ).model_dump()
    except Exception as exc:
        await telemetry.log_event(
            event_type="mcp_commit_learned_fact_error",
            agent_id=agent_id or "unscoped",
            success=False,
            details={"error": str(exc)},
        )
        return ToolResponse(
            success=False,
            error_code="COMMIT_FACT_FAILED",
            remediation_hint=(
                f"Provide a valid agent_id and rule_text (3-500 chars). Details: {exc}"
            ),
        ).model_dump()


@mcp.tool()
async def report_rule_outcome(
    agent_id: str,
    memory_id: str,
    succeeded: bool,
    prune_threshold: float = 0.3,
) -> dict[str, Any]:
    """Adaptive utility feedback: Report whether a recalled procedural rule succeeded or failed.

    Boosts the rule's utility score by +0.1 on success (capped at 2.0) or penalizes by -0.4 on
    failure. Automatically prunes the rule from Qdrant if its utility score drops below `prune_threshold`.

    Args:
        agent_id: Unique identifier of the calling agent.
        memory_id: Vector ID of the recalled procedural rule.
        succeeded: True if applying the rule succeeded; False if execution still failed.
        prune_threshold: Score threshold below which the rule is auto-pruned (default 0.3).
    """
    try:
        feedback = RuleFeedbackInput(
            agent_id=agent_id,
            memory_id=memory_id,
            succeeded=succeeded,
            prune_threshold=prune_threshold,
        )
        outcome = await get_engine().record_rule_outcome(
            memory_id=feedback.memory_id,
            agent_id=feedback.agent_id,
            succeeded=feedback.succeeded,
            prune_threshold=feedback.prune_threshold,
        )
        return ToolResponse(
            success=True,
            data=outcome.model_dump(),
        ).model_dump()
    except Exception as exc:
        await telemetry.log_event(
            event_type="mcp_report_rule_outcome_error",
            agent_id=agent_id or "unscoped",
            success=False,
            details={"error": str(exc)},
        )
        return ToolResponse(
            success=False,
            error_code="RULE_OUTCOME_FAILED",
            remediation_hint=(
                f"Ensure agent_id and memory_id are valid non-empty strings. Details: {exc}"
            ),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
