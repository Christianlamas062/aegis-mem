"""FastAPI Cloud API for multi-agent procedural memory recall, non-blocking reflection, and utility feedback."""

from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, FastAPI, status
from fastapi.responses import JSONResponse

from aegis_mem import __version__
from aegis_mem.core.engine import get_engine
from aegis_mem.core.reflector import distill_failure
from aegis_mem.core.telemetry import telemetry
from aegis_mem.schemas import (
    FailureReflectionInput,
    MemoryCreate,
    MemorySearchQuery,
    RuleFeedbackInput,
    ToolResponse,
)

app = FastAPI(
    title="aegis-mem Cloud API",
    version=__version__,
    description="Self-Evolving Procedural & Episodic Agent Memory Service (Mem0 + Qdrant)",
)


@app.get("/health", response_model=dict[str, Any])
async def health_check() -> dict[str, Any]:
    """Liveness and readiness probe."""
    return {
        "status": "healthy",
        "service": "aegis-mem",
        "version": __version__,
    }


@app.post("/v1/memories/recall", response_model=ToolResponse)
async def recall_memories(query: MemorySearchQuery) -> ToolResponse | JSONResponse:
    """Pre-flight recall of learned procedural rules scoped to `agent_id`."""
    try:
        rules = await get_engine().recall_context(query)
        return ToolResponse(
            success=True,
            data={"rules": [r.model_dump() for r in rules], "count": len(rules)},
        )
    except Exception as exc:
        await telemetry.log_event(
            event_type="api_recall_error",
            agent_id=query.agent_id,
            success=False,
            details={"error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ToolResponse(
                success=False,
                error_code="RECALL_ENGINE_ERROR",
                remediation_hint=f"Failed to query vector store: {exc}",
            ).model_dump(),
        )


@app.post(
    "/v1/memories/reflect",
    response_model=ToolResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reflect_on_failure(
    payload: FailureReflectionInput,
    background_tasks: BackgroundTasks,
) -> ToolResponse:
    """Enqueue cognitive distillation in the background so the calling agent is never blocked."""
    background_tasks.add_task(
        distill_failure,
        error_trace=payload.error_trace,
        input_context=payload.input_context,
        agent_id=payload.agent_id,
        tool_name=payload.tool_name,
        metadata=payload.metadata,
    )
    return ToolResponse(
        success=True,
        data={
            "status": "reflection_scheduled",
            "agent_id": payload.agent_id,
            "tool_name": payload.tool_name,
        },
    )


@app.post(
    "/v1/memories/commit",
    response_model=ToolResponse,
    status_code=status.HTTP_201_CREATED,
)
async def commit_memory(payload: MemoryCreate) -> ToolResponse | JSONResponse:
    """Explicitly commit a procedural rule to the agent's vector memory."""
    try:
        stored = await get_engine().store_rule(
            agent_id=payload.agent_id,
            rule_text=payload.rule_text,
            metadata={"category": payload.category, **payload.metadata},
            user_id=payload.user_id,
            run_id=payload.run_id,
        )
        return ToolResponse(
            success=True,
            data={"stored": stored, "agent_id": payload.agent_id},
        )
    except Exception as exc:
        await telemetry.log_event(
            event_type="api_commit_error",
            agent_id=payload.agent_id,
            success=False,
            details={"error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ToolResponse(
                success=False,
                error_code="COMMIT_ENGINE_ERROR",
                remediation_hint=f"Failed to persist procedural rule: {exc}",
            ).model_dump(),
        )


@app.post("/v1/memories/feedback", response_model=ToolResponse)
async def report_memory_feedback(
    payload: RuleFeedbackInput,
) -> ToolResponse | JSONResponse:
    """Update adaptive utility score for a procedural rule and auto-prune if below threshold."""
    try:
        outcome = await get_engine().record_rule_outcome(
            memory_id=payload.memory_id,
            agent_id=payload.agent_id,
            succeeded=payload.succeeded,
            prune_threshold=payload.prune_threshold,
        )
        return ToolResponse(
            success=True,
            data=outcome.model_dump(),
        )
    except Exception as exc:
        await telemetry.log_event(
            event_type="api_feedback_error",
            agent_id=payload.agent_id,
            success=False,
            details={"error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ToolResponse(
                success=False,
                error_code="FEEDBACK_ENGINE_ERROR",
                remediation_hint=f"Failed to record rule outcome: {exc}",
            ).model_dump(),
        )
