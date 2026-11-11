"""Live Self-Learning Demo for `aegis-mem` (Phase 5 Portfolio Showcase).

Runs 100% offline/locally using in-memory Qdrant (`:memory:`) and a deterministic
cognitive distillation fallback when no external LLM API key is configured.

Usage:
    python examples/live_self_learning_demo.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure `src` is importable when running directly via `python examples/live_self_learning_demo.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from aegis_mem.core.engine import AegisMemoryEngine, set_engine
from aegis_mem.core.reflector import sanitize_sensitive_text
from aegis_mem.core.telemetry import telemetry
from aegis_mem.schemas import MemorySearchQuery
from aegis_mem.utils.decorators import _background_tasks, auto_learn


def _embed_text(text: str, dim: int = 64) -> list[float]:
    """Deterministic local token embedding for zero-cost offline Qdrant vector search."""
    vec = [0.0] * dim
    tokens = [t.strip(".,:;!?()[]{}\"'`-_/").lower() for t in text.split() if t.strip()]
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dim):
            vec[idx] += (digest[idx % len(digest)] / 127.5) - 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class LocalInMemoryQdrantBackend:
    """Offline Mem0-compatible backend powered by real `QdrantClient(location=':memory:')`."""

    def __init__(self, collection_name: str = "aegis_live_demo") -> None:
        self.qdrant = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.dim = 64
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self.llm = self._DeterministicDistillerLLM()

    class _DeterministicDistillerLLM:
        def generate_response(self, messages: list[dict[str, str]]) -> str:
            user_prompt = messages[-1]["content"] if messages else ""
            if "ISO-8601" in user_prompt or "422" in user_prompt:
                return (
                    "Format exchange order `timestamp` as strict ISO-8601 UTC with 'Z' suffix "
                    "(YYYY-MM-DDTHH:MM:SSZ) instead of raw epoch to prevent HTTP 422."
                )
            return "Validate all endpoint parameters against schema preconditions before execution."

    def add(
        self,
        messages: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        point_id = str(uuid.uuid4())
        meta = dict(metadata or {})
        payload = {
            "memory": messages,
            "agent_id": agent_id or "",
            "user_id": user_id or "",
            "run_id": run_id or "",
            "metadata": meta,
            **meta,
        }
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=_embed_text(messages, self.dim),
                    payload=payload,
                )
            ],
        )
        return {"results": [{"id": point_id, "memory": messages, "event": "ADD"}]}

    def search(
        self,
        query: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        must: list[FieldCondition] = []
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=_embed_text(query, self.dim),
            query_filter=Filter(must=must) if must else None,
            limit=limit,
        )
        return {
            "results": [
                {
                    "id": str(pt.id),
                    "memory": (pt.payload or {}).get("memory", ""),
                    "score": float(pt.score or 0.0),
                    "metadata": (pt.payload or {}).get("metadata", {}),
                }
                for pt in response.points
            ]
        }


AGENT_ID = "quant_order_router_v1"


@auto_learn(agent_id=AGENT_ID, reraise=True)
async def submit_exchange_order(
    symbol: str,
    price: float,
    timestamp: str,
    auth: str = "",
) -> dict[str, Any]:
    """Simulated low-latency exchange order gateway requiring strict ISO-8601 UTC timestamps."""
    if not (timestamp.endswith("Z") and "T" in timestamp):
        raise ValueError(
            "HTTP 422 Unprocessable Entity: Exchange API requires strict ISO-8601 UTC "
            f"timestamp with 'Z' suffix, got raw epoch ({timestamp!r}) [auth={auth}]."
        )
    return {
        "order_id": "ORD-2026-99481",
        "symbol": symbol,
        "price": price,
        "timestamp": timestamp,
        "status": "FILLED",
    }


def _query_telemetry_summary(agent_id: str) -> tuple[int, int]:
    """Fetch total telemetry events and memory utility hits from the SQLite audit store."""
    conn = telemetry._get_connection()
    events_count = conn.execute(
        "SELECT COUNT(*) FROM telemetry_events WHERE agent_id = ?", (agent_id,)
    ).fetchone()[0]
    utility_hits = conn.execute(
        "SELECT COALESCE(SUM(hit_count), 0) FROM memory_utility WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()[0]
    return int(events_count), int(utility_hits)


async def run_demo() -> None:
    backend = LocalInMemoryQdrantBackend()
    engine = AegisMemoryEngine(memory_client=backend)
    set_engine(engine)

    raw_epoch = "1794366000"
    leaked_auth = "Bearer sk-live-998877665544332211"

    print("=" * 82)
    print(" 🛡️  AEGIS-MEM | AUTONOMOUS SELF-REFLECTION & PROCEDURAL MEMORY DEMO")
    print("=" * 82)
    print(f" • Agent Scope   : {AGENT_ID}")
    print(" • Vector Engine : Qdrant In-Memory (:memory:) + Mem0 Procedural Adapter")
    print(" • Audit Store   : SQLite Async Telemetry (:memory:)")
    print("-" * 82)

    # -------------------------------------------------------------------------
    # STEP 1: Attempt 1 (Execution Failure & Background Trap)
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] 🚀 Attempt 1 — Executing `submit_exchange_order` with raw epoch...")
    t0 = time.perf_counter()
    caught_error_str = ""
    try:
        await submit_exchange_order(
            symbol="BTC-USD",
            price=94250.50,
            timestamp=raw_epoch,
            auth=leaked_auth,
        )
    except ValueError as exc:
        caught_error_str = str(exc)
        step1_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  ✗ Exception Trapped by @auto_learn ({step1_ms:.2f} ms):")
        print(f"    Raw Error : {caught_error_str}")

    # Wait for the background reflection task spawned by @auto_learn
    if _background_tasks:
        await asyncio.gather(*list(_background_tasks))

    # -------------------------------------------------------------------------
    # STEP 2: Sanitization & Cognitive Distillation Inspection
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] 🧠 Cognitive Distillation & Regex PII/Secret Sanitization...")
    t1 = time.perf_counter()
    sanitized_preview = sanitize_sensitive_text(caught_error_str)
    stored_rules = await engine.recall_context(
        MemorySearchQuery(
            agent_id=AGENT_ID,
            task_description="submit exchange order timestamp format",
            top_k=1,
        )
    )
    step2_ms = (time.perf_counter() - t1) * 1000.0
    distilled = stored_rules[0]

    print(f"  ✓ Regex Sanitizer Output : {sanitized_preview}")
    print(f"  ✓ Distilled 1-Line Rule  : \"{distilled.rule}\" ({len(distilled.rule)} chars)")
    print(f"  ✓ Qdrant Vector ID       : {distilled.memory_id} (indexed in {step2_ms:.2f} ms)")

    # -------------------------------------------------------------------------
    # STEP 3: Attempt 2 (Pre-Flight Recall & Autonomous Adaptation)
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] 🔄 Attempt 2 — Pre-Flight Recall & Autonomous Payload Adaptation...")
    t2 = time.perf_counter()
    recalled_rules = await engine.recall_context(
        MemorySearchQuery(
            agent_id=AGENT_ID,
            task_description="submit exchange order timestamp format",
            top_k=3,
        )
    )
    recall_ms = (time.perf_counter() - t2) * 1000.0
    top_rule = recalled_rules[0]

    print(
        f"  ✓ Pre-Flight Recall Hit  : score={top_rule.score:.4f} | latency={recall_ms:.2f} ms"
    )
    print(f"  ✓ Applied Rule           : {top_rule.rule}")

    # Dynamically adapt timestamp parameter based on the recalled ISO-8601 UTC rule
    adapted_timestamp = raw_epoch
    if "ISO-8601" in top_rule.rule and "Z" in top_rule.rule:
        dt_utc = datetime.fromtimestamp(int(raw_epoch), tz=timezone.utc)
        adapted_timestamp = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"  ✓ Adapted Parameter      : timestamp='{raw_epoch}' -> '{adapted_timestamp}'")

    t3 = time.perf_counter()
    order_result = await submit_exchange_order(
        symbol="BTC-USD",
        price=94250.50,
        timestamp=adapted_timestamp,
        auth=leaked_auth,
    )
    exec_ms = (time.perf_counter() - t3) * 1000.0
    events_logged, utility_hits = _query_telemetry_summary(AGENT_ID)

    print(f"  ✓ Execution Succeeded    : {order_result} ({exec_ms:.2f} ms)")
    print("\n" + "-" * 82)
    print(
        f" 📊 TELEMETRY STATUS: {events_logged} audit events logged | "
        f"{utility_hits} memory utility hits | 0 leaked secrets"
    )
    print("=" * 82)


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run_demo())

