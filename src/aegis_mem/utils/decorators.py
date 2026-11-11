"""Asynchronous `@auto_learn` (and alias `@learn_from_exception`) decorator.

Wraps any async or sync Python callable; when an exception is raised, captures the
stacktrace and call arguments, fires `reflector.distill_failure` in the background
(`asyncio.create_task`), and either re-raises or returns a structured `ToolResponse`.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
from typing import Any, Callable, TypeVar

from aegis_mem.core.engine import AegisMemoryEngine
from aegis_mem.core.reflector import distill_failure
from aegis_mem.schemas import ToolResponse

F = TypeVar("F", bound=Callable[..., Any])

_background_tasks: set[asyncio.Task[Any]] = set()


def _format_call_context(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Serialize function invocation arguments into a concise context string."""
    parts: list[str] = []
    if args:
        parts.append(f"args={args!r}")
    if kwargs:
        parts.append(f"kwargs={kwargs!r}")
    joined = ", ".join(parts) if parts else "no_args"
    return f"{func_name}({joined})"


def auto_learn(
    agent_id: str,
    *,
    reraise: bool = True,
    engine: AegisMemoryEngine | None = None,
) -> Callable[[F], Callable[..., Any]]:
    """Decorator that intercepts exceptions, triggers background cognitive distillation, and re-raises or handles cleanly.

    Args:
        agent_id: Mandatory agent scope to associate learned procedural rules with.
        reraise: If True (default), re-raises the original exception after scheduling background
            reflection. If False, catches the exception and returns a `ToolResponse(success=False, ...)`.
        engine: Optional explicit `AegisMemoryEngine` instance (useful for testing).
    """
    if not agent_id or not agent_id.strip():
        raise ValueError("auto_learn requires a non-empty agent_id")

    def decorator(func: F) -> Callable[..., Any]:
        is_coro = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if is_coro:
                    return await func(*args, **kwargs)
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:
                tb_str = traceback.format_exc()
                ctx_str = _format_call_context(func.__name__, args, kwargs)
                task = asyncio.create_task(
                    distill_failure(
                        error_trace=tb_str,
                        input_context=ctx_str,
                        agent_id=agent_id,
                        tool_name=func.__name__,
                        engine=engine,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

                if reraise:
                    raise
                return ToolResponse(
                    success=False,
                    error_code=type(exc).__name__,
                    remediation_hint=(
                        f"Execution failed in '{func.__name__}': {exc}. "
                        "A self-reflection task has been scheduled to distill a preventative rule."
                    ),
                )

        return async_wrapper

    return decorator


# Alias matching the file tree description (@learn_from_exception)
learn_from_exception = auto_learn

