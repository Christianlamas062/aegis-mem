"""Asynchronous procedural & episodic memory engine powered by Mem0 + Qdrant."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mem0 import Memory

from aegis_mem.config import settings
from aegis_mem.core.telemetry import telemetry
from aegis_mem.schemas import (
    MemoryCreate,
    MemorySearchQuery,
    ProceduralRuleOutput,
    RuleOutcomeResult,
)


class AegisMemoryEngine:
    """Asynchronous procedural & episodic memory engine powered by Mem0 + Qdrant."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        memory_client: Any | None = None,
    ) -> None:
        if memory_client is not None:
            self._memory = memory_client
        else:
            effective_config = config or settings.build_mem0_config()
            self._memory = Memory.from_config(effective_config)

    @staticmethod
    def _enforce_scope(
        agent_id: str | None,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Ensure at least one scope identifier is present (No Unbounded Context)."""
        cleaned_agent = (agent_id or "").strip()
        cleaned_user = (user_id or "").strip()
        cleaned_run = (run_id or "").strip()
        if not (cleaned_agent or cleaned_user or cleaned_run):
            raise ValueError(
                "Unbounded memory operation forbidden: agent_id, user_id, or run_id must be provided."
            )
        return cleaned_agent or cleaned_user or cleaned_run

    async def store_rule(
        self,
        agent_id: str,
        rule_text: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a distilled procedural rule scoped to a specific agent."""
        scope_id = self._enforce_scope(agent_id, user_id, run_id)
        payload = MemoryCreate(
            agent_id=scope_id,
            user_id=user_id,
            run_id=run_id,
            rule_text=rule_text,
            category=(metadata or {}).get("category", "error_remediation"),
            metadata=metadata or {},
        )
        merged_metadata: dict[str, Any] = {
            "category": payload.category,
            "type": "procedural_rule",
            **payload.metadata,
        }

        start = time.perf_counter()
        call_kwargs: dict[str, Any] = {
            "messages": payload.rule_text,
            "agent_id": payload.agent_id,
            "metadata": merged_metadata,
        }
        if payload.user_id:
            call_kwargs["user_id"] = payload.user_id
        if payload.run_id:
            call_kwargs["run_id"] = payload.run_id

        result = await asyncio.to_thread(self._memory.add, **call_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        await telemetry.log_event(
            event_type="store_rule",
            agent_id=payload.agent_id,
            latency_ms=elapsed_ms,
            success=True,
            details={"rule_text": payload.rule_text, "category": payload.category},
        )
        if isinstance(result, dict):
            return result
        return {"results": result}

    async def store_procedural_rule(
        self, agent_id: str, rule_text: str, category: str = "error_remediation"
    ) -> dict[str, Any]:
        """Alias compatible with Phase 5 reference implementation."""
        return await self.store_rule(
            agent_id=agent_id,
            rule_text=rule_text,
            metadata={"category": category, "type": "procedural_rule"},
        )

    async def recall_rules(
        self,
        agent_id: str,
        task_query: str,
        limit: int = 5,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> list[ProceduralRuleOutput]:
        """Retrieve relevant learned rules before executing a tool call."""
        scope_id = self._enforce_scope(agent_id, user_id, run_id)
        search_query = MemorySearchQuery(
            agent_id=scope_id,
            user_id=user_id,
            run_id=run_id,
            task_description=task_query,
            top_k=limit,
        )
        return await self.recall_context(search_query)

    async def recall_context(
        self, query: MemorySearchQuery
    ) -> list[ProceduralRuleOutput]:
        """Retrieve relevant learned rules using a validated `MemorySearchQuery`."""
        self._enforce_scope(query.agent_id, query.user_id, query.run_id)
        start = time.perf_counter()

        search_kwargs: dict[str, Any] = {
            "query": query.task_description,
            "agent_id": query.agent_id,
            "limit": query.top_k,
        }
        if query.user_id:
            search_kwargs["user_id"] = query.user_id
        if query.run_id:
            search_kwargs["run_id"] = query.run_id

        raw_results = await asyncio.to_thread(self._memory.search, **search_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        items = (
            raw_results.get("results", [])
            if isinstance(raw_results, dict)
            else (raw_results or [])
        )
        outputs = [
            ProceduralRuleOutput(
                memory_id=str(item.get("id", "")),
                rule=str(item.get("memory", item.get("text", ""))),
                score=float(item.get("score", 0.0)),
                metadata=item.get("metadata") or {},
            )
            for item in items
            if isinstance(item, dict)
        ]

        await telemetry.log_event(
            event_type="recall_rules",
            agent_id=query.agent_id,
            latency_ms=elapsed_ms,
            success=True,
            details={"query": query.task_description, "count": len(outputs)},
        )
        for out in outputs:
            if out.memory_id:
                await telemetry.record_utility(out.memory_id, query.agent_id, succeeded=True)

        return outputs

    async def prune_obsolete(
        self, memory_id: str, agent_id: str = "system_pruner"
    ) -> dict[str, Any]:
        """Delete an obsolete or contradicted procedural rule by its vector ID."""
        cleaned_id = memory_id.strip()
        if not cleaned_id:
            raise ValueError("memory_id cannot be empty")
        start = time.perf_counter()
        await asyncio.to_thread(self._memory.delete, memory_id=cleaned_id)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        await telemetry.log_event(
            event_type="prune_obsolete",
            agent_id=agent_id,
            latency_ms=elapsed_ms,
            success=True,
            details={"memory_id": cleaned_id},
        )
        return {"deleted": True, "memory_id": cleaned_id}

    async def record_rule_outcome(
        self,
        memory_id: str,
        agent_id: str,
        succeeded: bool,
        prune_threshold: float = 0.3,
    ) -> RuleOutcomeResult:
        """Update adaptive utility score (+0.1 on success capped at 2.0; -0.4 on failure) and auto-prune if < threshold."""
        scope_id = self._enforce_scope(agent_id)
        cleaned_mem_id = memory_id.strip()
        if not cleaned_mem_id:
            raise ValueError("memory_id cannot be empty")

        start = time.perf_counter()
        updated_score = await telemetry.update_rule_outcome(
            memory_id=cleaned_mem_id,
            agent_id=scope_id,
            succeeded=succeeded,
        )
        pruned = False
        if updated_score < prune_threshold:
            await self.prune_obsolete(memory_id=cleaned_mem_id, agent_id=scope_id)
            pruned = True
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            await telemetry.log_event(
                event_type="stale_rule_auto_pruned",
                agent_id=scope_id,
                latency_ms=elapsed_ms,
                success=True,
                details={
                    "memory_id": cleaned_mem_id,
                    "utility_score": updated_score,
                    "prune_threshold": prune_threshold,
                },
            )
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            await telemetry.log_event(
                event_type="rule_outcome_recorded",
                agent_id=scope_id,
                latency_ms=elapsed_ms,
                success=True,
                details={
                    "memory_id": cleaned_mem_id,
                    "succeeded": succeeded,
                    "utility_score": updated_score,
                },
            )

        return RuleOutcomeResult(
            memory_id=cleaned_mem_id,
            agent_id=scope_id,
            succeeded=succeeded,
            utility_score=updated_score,
            pruned=pruned,
        )


_default_engine: AegisMemoryEngine | None = None


def get_engine() -> AegisMemoryEngine:
    """Lazy singleton accessor so importing modules does not trigger network calls."""
    global _default_engine
    if _default_engine is None:
        _default_engine = AegisMemoryEngine()
    return _default_engine


def set_engine(engine: AegisMemoryEngine | None) -> None:
    """Override the active engine instance (used by unit tests and custom DI)."""
    global _default_engine
    _default_engine = engine
