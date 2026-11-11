# Project Rules: aegis-mem (Self-Evolving Agent Memory)

## 1. Architectural Constraints
- **Strict Layer Separation**: `core/` handles Mem0, Qdrant, and reflection logic. `interfaces/` (FastAPI & FastMCP) must ONLY call `core/engine.py` and never instantiate `mem0.Memory` directly.
- **Async-First Execution**: All I/O operations (FastAPI routes, MCP tools, telemetry logging, and background reflections) must be asynchronous (`async def`). Wrap synchronous `mem0ai` calls using `asyncio.to_thread()` to prevent blocking the event loop.
- **No Unbounded Context**: Every memory operation MUST enforce `agent_id`, `user_id`, or `run_id` scoping. Global unscoped searches are strictly forbidden.

## 2. Code Style & Data Contracts
- **Typing & Validation**: Python 3.12+ strict type hints (`PEP 484`) required on all signatures. All payloads entering or leaving `core/` must be validated via Pydantic v2 models defined in `src/aegis_mem/schemas.py`.
- **Self-Reflection Discipline**: `core/reflector.py` must sanitize credentials, API keys, and PII via regex before passing error tracebacks to any LLM for lesson distillation.
- **Error Handling**: Never raise unhandled exceptions inside MCP tools. Catch errors, log to `telemetry.py`, and return a structured `ToolResponse(success=False, error_code=..., remediation_hint=...)`.

## 3. Testing Requirements
- Every new feature in `core/` requires a corresponding `pytest` unit test in `tests/`.
- Tests must use local in-memory Qdrant (`location=":memory:"`) and mocked LLM responses so the test suite runs offline in <3 seconds with zero API cost.

