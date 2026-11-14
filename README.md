# 🛡️ aegis-mem: Self-Evolving Procedural Agent Memory
[![M8ven Score](https://m8ven.ai/badge/mcp/christianlamas062-aegis-mem-ah76iy)](https://m8ven.ai/mcp/christianlamas062-aegis-mem-ah76iy?s=readme)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-emerald.svg)](https://github.com/jlowin/fastmcp)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-dc244c.svg)](https://qdrant.tech/)

**`aegis-mem`** is a production-grade procedural and episodic memory engine powered by **Mem0** and **Qdrant**. Rather than storing passive chat transcripts, `aegis-mem` intercepts runtime exceptions, scrubs credentials/PII via regex, deduplicates error storms with a pre-LLM SHA-256 circuit breaker, distills failures into deterministic **1-line procedural rules ($\le 200$ chars)**, and injects them during pre-flight recall so autonomous agents never repeat the same mistake twice.

---

## 🔄 Autonomous Self-Reflection Loop

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Autonomous Agent (@auto_learn)
    participant API as External Tool / Exchange API
    participant Reflector as Cognitive Reflector (reflector.py)
    participant Engine as AegisMemoryEngine (Mem0 + Qdrant)
    participant Audit as SQLite Telemetry (telemetry.py)

    Note over Agent,API: Step 1 — Attempt 1 (Execution & Trap)
    Agent->>API: submit_exchange_order(timestamp="1794366000", auth="Bearer <SESSION_TOKEN>")
    API-->>Agent: HTTP 422 Unprocessable Entity (Requires ISO-8601 UTC 'Z')
    Agent--)Reflector: asyncio.create_task(distill_failure(traceback, context))

    Note over Reflector,Engine: Step 2 — Zero-Trust Regex Sanitization & SHA-256 Circuit Breaker
    Reflector->>Reflector: sanitize_sensitive_text() -> "Bearer [REDACTED_TOKEN]"
    Reflector->>Reflector: compute_error_fingerprint() -> Check 300s TTL Dedup Cache
    Reflector->>Reflector: Synthesize 1-line deterministic rule (<= 200 chars)

    Note over Reflector,Audit: Step 3 — Evolutionary Vector Commit
    Reflector->>Engine: store_rule(agent_id="quant_order_router_v1", rule_text=...)
    Engine->>Audit: log_event("store_rule", latency_ms)

    Note over Agent,API: Step 4 — Pre-Flight Recall & Autonomous Adaptation (Attempt 2)
    Agent->>Engine: recall_context(agent_id="quant_order_router_v1", task="timestamp format")
    Engine-->>Agent: ProceduralRuleOutput(rule="Format timestamp as ISO-8601 UTC 'Z'...", score=0.70)
    Engine->>Audit: record_utility(memory_id, succeeded=True)
    Agent->>API: submit_exchange_order(timestamp="2026-11-11T03:00:00Z")
    API-->>Agent: 200 OK {"status": "FILLED", "order_id": "ORD-2026-99481"}
```

---

## 🧠 Staff-Grade Reliability Mechanisms

1. **Zero-Trust Regex PII & Secret Sanitizer (`core/reflector.py`)**:
   Before any stacktrace or request context reaches an LLM or vector index, `sanitize_sensitive_text()` strips `Bearer` tokens, `sk-*` API keys, AWS `AKIA/ASIA` credentials, database connection string passwords (`postgresql://user:pass@host`), and email addresses.
2. **Pre-LLM SHA-256 Deduplication Circuit Breaker (`core/reflector.py`)**:
   During downstream outages (e.g., an HTTP 502 Bad Gateway storm), naive reflection loops burn thousands of tokens distilling the exact same error. `compute_error_fingerprint()` normalizes volatile UUIDs, timestamps, hex addresses, and request IDs to compute a deterministic hash `sha256(f"{agent_id}:{exception_type}:{normalized_sanitized_error}")` backed by a configurable 300-second TTL cache (`_dedup_cache`). Duplicate exceptions within the cooldown window bypass the LLM and Qdrant completely (`0 tokens consumed`) and log a `"circuit_breaker_dedup_hit"` audit event.
3. **Adaptive Ebbinghaus / Utility Decay & Auto-Pruning (`core/engine.py` & `core/telemetry.py`)**:
   Every procedural rule starts at a base utility score of `1.0`. Agents report execution outcomes via `record_rule_outcome()` (exposed on MCP as `report_rule_outcome` and REST as `POST /v1/memories/feedback`):
   - **Positive reinforcement (`succeeded=True`)**: boosts utility score by `+0.1` (capped at `2.0`).
   - **Negative decay (`succeeded=False`)**: penalizes utility score by `-0.4` when a recalled rule fails to prevent an error.
   - **Automatic Vector Pruning**: once a stale or hallucinated rule's score drops below `prune_threshold` (`< 0.3`), `AegisMemoryEngine` automatically invokes `prune_obsolete(memory_id)` to purge it from Qdrant and records a `"stale_rule_auto_pruned"` telemetry event.

---

## 📊 Quantitative Benchmark: Baseline Stateless vs. Aegis-Mem Agent

Reproduce locally via `python benchmarks/run_benchmark.py`.

Workload: **20 recurring API execution iterations** (10 schema validation + 10 rate-limit tasks) + **10-event HTTP 502 Bad Gateway storm**.

| Metric | Baseline Stateless Agent | Aegis-Mem Agent (`@auto_learn` + Circuit Breaker) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Recurring Task Failures (20 runs)** | 20 / 20 (100% fail rate) | 2 / 20 (only 1st occurrence per fault class) | **-90% recurring failures** |
| **Total Exceptions (incl. 10x 502 Storm)** | 30 | 12 | **-60.0% total retries** |
| **Estimated Prompt Tokens Consumed** | 22,200 tokens | 5,624 tokens | **-74.7% token cost** |
| **Duplicate LLM Reflections Blocked** | 0 / 10 (0%) | 9 / 10 (90% blocked via SHA-256 TTL) | **9 redundant LLM calls saved** |
| **Mean Pre-Flight Recall Latency** | N/A (No memory) | 2.23 ms | **Sub-5ms Qdrant `:memory:`** |

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Run the Live Self-Learning Demo & Quantitative Benchmark (100% Offline)

```bash
python examples/live_self_learning_demo.py
python benchmarks/run_benchmark.py
```

### 3. Run the Test Suite

All 10 unit and integration tests run offline against `QdrantClient(location=":memory:")` with zero external API cost:

```bash
pytest -v --no-header
```

### 4. Configure as an IDE / Claude Desktop MCP Server

Add the following entry to your `claude_desktop_config.json` or IDE MCP settings (`.cursor/mcp.json` / Antigravity MCP config):

```json
{
  "mcpServers": {
    "aegis-mem": {
      "command": "python",
      "args": ["-m", "aegis_mem.interfaces.mcp_server"],
      "env": {
        "AEGIS_QDRANT_IN_MEMORY": "true",
        "AEGIS_COLLECTION_NAME": "aegis_procedural_rules",
        "AEGIS_LLM_PROVIDER": "openai",
        "AEGIS_LLM_MODEL": "gpt-4o-mini",
        "AEGIS_LLM_API_KEY": "YOUR_OPENAI_API_KEY"
      }
    }
  }
}
```

Or using `uv`:

```json
{
  "mcpServers": {
    "aegis-mem": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/aegis-mem",
        "run",
        "python",
        "-m",
        "aegis_mem.interfaces.mcp_server"
      ]
    }
  }
}
```

---

## 🐳 Production Deployment (Docker Compose)

Launch Qdrant and the `aegis-mem` FastAPI service on port `8000`:

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health
```
