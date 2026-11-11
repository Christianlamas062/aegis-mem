"""Unit tests for `AegisMemoryEngine` retention, scoping, conflict resolution, pruning, and utility decay."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aegis_mem.core.engine import AegisMemoryEngine
from aegis_mem.schemas import MemorySearchQuery


@pytest.mark.asyncio
async def test_store_and_recall_scoped_rules(in_memory_engine: AegisMemoryEngine) -> None:
    """Verify rules stored for one agent are recalled for that agent and isolated from others."""
    await in_memory_engine.store_rule(
        agent_id="billing-agent",
        rule_text="Always pass currency in ISO-4217 uppercase (e.g., USD, EUR) to /v1/charges.",
        metadata={"category": "schema_constraint", "endpoint": "/v1/charges"},
    )
    await in_memory_engine.store_rule(
        agent_id="inventory-agent",
        rule_text="Warehouse SKU queries require a non-empty region_code parameter.",
        metadata={"category": "schema_constraint", "endpoint": "/v1/skus"},
    )

    billing_rules = await in_memory_engine.recall_rules(
        agent_id="billing-agent",
        task_query="Create a charge on /v1/charges in USD currency",
        limit=5,
    )
    assert len(billing_rules) == 1
    assert "ISO-4217" in billing_rules[0].rule
    assert billing_rules[0].metadata.get("endpoint") == "/v1/charges"

    # Ensure inventory-agent cannot see billing-agent's rules
    inventory_rules = await in_memory_engine.recall_rules(
        agent_id="inventory-agent",
        task_query="Create a charge on /v1/charges in USD currency",
        limit=5,
    )
    assert len(inventory_rules) == 1
    assert "Warehouse SKU" in inventory_rules[0].rule


@pytest.mark.asyncio
async def test_conflict_resolution_updates_existing_endpoint_rule(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify updating a rule for the same agent + endpoint overwrites obsolete vector instead of duplicating."""
    first = await in_memory_engine.store_rule(
        agent_id="sql-agent",
        rule_text="Use v1_users table for customer lookups.",
        metadata={"endpoint": "users_lookup"},
    )
    first_id = first["results"][0]["id"]

    second = await in_memory_engine.store_rule(
        agent_id="sql-agent",
        rule_text="v1_users is deprecated; always query v2_customers table with tenant_id.",
        metadata={"endpoint": "users_lookup"},
    )
    second_id = second["results"][0]["id"]

    assert first_id == second_id
    recalled = await in_memory_engine.recall_rules(
        agent_id="sql-agent",
        task_query="Query customer table",
        limit=5,
    )
    assert len(recalled) == 1
    assert "v2_customers" in recalled[0].rule


@pytest.mark.asyncio
async def test_prune_obsolete_removes_rule(in_memory_engine: AegisMemoryEngine) -> None:
    """Verify `prune_obsolete` deletes a rule by ID from Qdrant."""
    stored = await in_memory_engine.store_rule(
        agent_id="ops-agent",
        rule_text="Temporary maintenance window rule on /v1/deploy.",
    )
    mem_id = stored["results"][0]["id"]

    before = await in_memory_engine.recall_rules("ops-agent", "deploy service")
    assert len(before) == 1

    prune_resp = await in_memory_engine.prune_obsolete(mem_id, agent_id="ops-agent")
    assert prune_resp["deleted"] is True

    after = await in_memory_engine.recall_rules("ops-agent", "deploy service")
    assert len(after) == 0


@pytest.mark.asyncio
async def test_unbounded_context_is_rejected(in_memory_engine: AegisMemoryEngine) -> None:
    """Verify unscoped calls (empty agent_id) are strictly rejected."""
    with pytest.raises(ValueError, match="Unbounded memory operation forbidden"):
        await in_memory_engine.recall_rules(agent_id="   ", task_query="any query")

    with pytest.raises(ValidationError):
        MemorySearchQuery(agent_id="   ", task_description="any query")


@pytest.mark.asyncio
async def test_utility_decay_auto_prunes_failing_rule(
    in_memory_engine: AegisMemoryEngine,
) -> None:
    """Verify two consecutive negative outcomes drop score < 0.3 (1.0 -> 0.6 -> 0.2) and purge the vector from Qdrant."""
    stored = await in_memory_engine.store_rule(
        agent_id="decay-agent",
        rule_text="Outdated routing rule: send all /v1/payments requests to legacy-gateway-01.",
        metadata={"endpoint": "/v1/payments"},
    )
    mem_id = stored["results"][0]["id"]

    # First negative outcome: 1.0 - 0.4 = 0.6 (still >= 0.3, not pruned yet)
    outcome_1 = await in_memory_engine.record_rule_outcome(
        memory_id=mem_id,
        agent_id="decay-agent",
        succeeded=False,
        prune_threshold=0.3,
    )
    assert outcome_1.utility_score == pytest.approx(0.6)
    assert outcome_1.pruned is False

    still_present = await in_memory_engine.recall_rules("decay-agent", "payments routing")
    assert len(still_present) == 1

    # Second negative outcome: 0.6 - 0.4 = 0.2 (< 0.3 -> automatically pruned from Qdrant!)
    outcome_2 = await in_memory_engine.record_rule_outcome(
        memory_id=mem_id,
        agent_id="decay-agent",
        succeeded=False,
        prune_threshold=0.3,
    )
    assert outcome_2.utility_score == pytest.approx(0.2)
    assert outcome_2.pruned is True

    after_prune = await in_memory_engine.recall_rules("decay-agent", "payments routing")
    assert len(after_prune) == 0
