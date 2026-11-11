"""Pydantic v2 data contracts for all inputs and outputs across core and interfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScopedContext(BaseModel):
    """Base model enforcing strict agent/user/run scoping (No Unbounded Context)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=1, max_length=128, description="Scoped identifier for the executing agent")
    user_id: str | None = Field(default=None, max_length=128, description="Optional user scope")
    run_id: str | None = Field(default=None, max_length=128, description="Optional execution run scope")

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("agent_id cannot be blank or whitespace only")
        return cleaned


class MemoryCreate(ScopedContext):
    """Contract for explicitly committing a learned procedural rule or fact."""

    rule_text: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Deterministic procedural rule or operational constraint",
    )
    category: str = Field(
        default="error_remediation",
        max_length=64,
        description="Rule taxonomy (e.g., error_remediation, schema_constraint, rate_limit)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured attributes (endpoint, tool_name, etc.)",
    )


class FailureReflectionInput(ScopedContext):
    """Contract for triggering cognitive distillation from an execution failure."""

    error_trace: str = Field(
        ...,
        min_length=1,
        description="Raw exception message, HTTP error payload, or Python traceback",
    )
    input_context: str = Field(
        ...,
        min_length=1,
        description="Task intent, arguments, or request payload that triggered the failure",
    )
    tool_name: str | None = Field(
        default=None,
        max_length=128,
        description="Name of the tool, function, or endpoint that failed",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchQuery(ScopedContext):
    """Contract for pre-flight procedural memory retrieval."""

    task_description: str = Field(
        ...,
        min_length=1,
        description="The intent, query, or upcoming action the agent is about to execute",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of procedural rules to retrieve",
    )


class ProceduralRuleOutput(BaseModel):
    """Standardized procedural rule returned from vector recall."""

    model_config = ConfigDict(extra="ignore")

    memory_id: str = Field(..., description="Unique identifier in the vector store")
    rule: str = Field(..., description="Distilled 1-line procedural rule")
    score: float = Field(default=0.0, description="Hybrid relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Rule metadata")


class ReflectionResult(BaseModel):
    """Output contract produced by `reflector.distill_failure`."""

    agent_id: str
    sanitized_error: str
    distilled_rule: str = Field(..., max_length=200)
    memory_id: str | None = None
    stored: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResponse(BaseModel):
    """Structured envelope returned by all MCP tools and API endpoints."""

    success: bool = Field(..., description="Whether the operation completed without error")
    data: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Result payload when success=True"
    )
    error_code: str | None = Field(
        default=None, description="Machine-readable error code when success=False"
    )
    remediation_hint: str | None = Field(
        default=None, description="Actionable instruction for the calling agent on error"
    )

