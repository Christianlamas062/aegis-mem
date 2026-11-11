"""Offline pytest fixtures backed by real in-memory Qdrant (`location=':memory:'`) and deterministic mock LLMs."""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from aegis_mem.core.engine import AegisMemoryEngine, set_engine
from aegis_mem.core.reflector import clear_dedup_cache


def _deterministic_embed(text: str, dim: int = 32) -> list[float]:
    """Generate a normalized deterministic embedding vector from text tokens for offline semantic similarity."""
    vec = [0.0] * dim
    tokens = [t.strip(".,:;!?()[]{}\"'").lower() for t in text.split() if t.strip()]
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            val = (digest[i % len(digest)] / 127.5) - 1.0
            vec[i] += val
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class InMemoryQdrantMem0Adapter:
    """Mem0-compatible adapter backed by `QdrantClient(location=':memory:')` and mocked LLM distillation."""

    def __init__(self, collection_name: str = "test_aegis_rules") -> None:
        self.qdrant = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.vector_dim = 32
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
        self.llm = self._MockLLM()

    class _MockLLM:
        def generate_response(self, messages: list[dict[str, str]]) -> str:
            user_msg = messages[-1]["content"] if messages else ""
            if "422" in user_msg or "date" in user_msg.lower():
                return "Always format `start_date` as ISO-8601 (YYYY-MM-DD) when calling /v1/invoices to avoid HTTP 422."
            if "429" in user_msg or "rate" in user_msg.lower():
                return "Enforce max 10 requests/sec and exponential backoff on /v1/search to avoid HTTP 429."
            return "Validate all required schema parameters before invoking the external endpoint."

    def add(
        self,
        messages: str | list[dict[str, str]],
        agent_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rule_text = messages if isinstance(messages, str) else str(messages)
        vector = _deterministic_embed(rule_text, self.vector_dim)

        # Check if an existing rule on the same endpoint/category contradicts or should be updated in-place
        point_id = str(uuid.uuid4())
        meta = dict(metadata or {})
        if agent_id:
            meta["agent_id"] = agent_id
        if user_id:
            meta["user_id"] = user_id
        if run_id:
            meta["run_id"] = run_id

        if agent_id and "endpoint" in meta:
            existing, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                        FieldCondition(key="endpoint", match=MatchValue(value=meta["endpoint"])),
                    ]
                ),
                limit=1,
            )
            if existing:
                point_id = str(existing[0].id)

        payload = {
            "memory": rule_text,
            "agent_id": agent_id or "",
            "user_id": user_id or "",
            "run_id": run_id or "",
            "metadata": meta,
            **meta,
        }
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return {"results": [{"id": point_id, "memory": rule_text, "event": "ADD"}]}

    def search(
        self,
        query: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        vector = _deterministic_embed(query, self.vector_dim)
        must_conditions: list[FieldCondition] = []
        if agent_id:
            must_conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if user_id:
            must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if run_id:
            must_conditions.append(FieldCondition(key="run_id", match=MatchValue(value=run_id)))

        query_filter = Filter(must=must_conditions) if must_conditions else None
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )
        hits = response.points
        results = [
            {
                "id": str(hit.id),
                "memory": (hit.payload or {}).get("memory", ""),
                "score": float(hit.score or 0.0),
                "metadata": (hit.payload or {}).get("metadata", {}),
            }
            for hit in hits
        ]
        return {"results": results}

    def delete(self, memory_id: str) -> dict[str, Any]:
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=[memory_id],
        )
        return {"deleted": True, "id": memory_id}


@pytest.fixture()
def in_memory_engine() -> AegisMemoryEngine:
    """Provide an `AegisMemoryEngine` wired to an in-memory Qdrant instance (`:memory:`)."""
    clear_dedup_cache()
    adapter = InMemoryQdrantMem0Adapter()
    engine = AegisMemoryEngine(memory_client=adapter)
    set_engine(engine)
    yield engine
    clear_dedup_cache()
    set_engine(None)
