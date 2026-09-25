"""Expose Laya's published shortlist helper through the standalone MCP server."""

import time
from typing import Any, Callable, Sequence

from laya.mcp.device import agent_device, router_agent
from laya.mcp.tools import ToolError, validate_model, validate_questions, validate_state
from laya.shortlist import DEFAULT_SHORTLIST_K, embed_fn_from_agent, predict_shortlist


def _agent_for(router: Any, name: str) -> Any:
    agent = router_agent(router, name)
    if agent is not None:
        return agent
    load = getattr(router, "load", None)
    if load is None:
        raise ToolError("models_not_ready", f"checkpoint {name!r} is not loaded")
    return load(name)


def laya_shortlist(
    state: Any,
    questions: Any,
    model: Any = "auto",
    k: Any = None,
    *,
    router: Any = None,
    agent: Any = None,
    embed_fn: Callable[[Sequence[str]], Any] | None = None,
) -> dict:
    """Narrow large choice questions using the answering checkpoint's embeddings."""
    state = validate_state(state)
    questions = validate_questions(questions)
    model = validate_model(model)
    if k is None:
        k = DEFAULT_SHORTLIST_K
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ToolError("invalid_k", f"k must be a positive integer, got {k!r}")

    if model == "auto":
        if router is None:
            raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
        if not hasattr(router, "route"):
            raise ToolError("internal_error", "router has no route() method")
        decision = router.route(state, questions)
        if isinstance(decision, dict):
            routed = decision.get("model")
            routing = {name: decision.get(name) for name in ("model", "repo", "reason")}
        else:
            routed = getattr(decision, "model", None)
            routing = {name: getattr(decision, name, None) for name in ("model", "repo", "reason")}
        if not isinstance(routed, str) or not routed:
            raise ToolError("internal_error", "router.route returned no model")
        target, kwargs = router, {"model": routed}
        embedding_agent = _agent_for(router, routed)
    elif agent is not None:
        routing = {"model": model, "repo": None, "reason": "explicit model"}
        target, kwargs, embedding_agent = agent, {}, agent
    else:
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        routing = {"model": model, "repo": None, "reason": "explicit model"}
        target, kwargs = router, {"model": model}
        embedding_agent = _agent_for(router, model)

    if embed_fn is None:
        try:
            embed_fn = embed_fn_from_agent(embedding_agent)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ToolError("models_not_ready", f"cannot build shortlist embeddings: {exc}") from exc

    started = time.perf_counter()
    result = predict_shortlist(target, state, questions, embed_fn, k=k, **kwargs)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ToolError("internal_error", "predict returned non-object answers")
    output = {
        "answers": result["answers"],
        "routing": routing,
        "shortlist": result.get("shortlist") or {},
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    device = agent_device(embedding_agent)
    if device:
        output["device"] = device
    return output
