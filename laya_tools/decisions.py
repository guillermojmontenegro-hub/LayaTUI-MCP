"""Choose among concrete actions using the official Laya PyPI package."""

import time
from typing import Any, Callable, Sequence

from laya.mcp.device import agent_device
from laya.mcp.tools import ToolError, laya_predict

from .shortlist import laya_shortlist


def laya_choose_action(
    goal: Any,
    observation: Any,
    actions: Any,
    history: Any = None,
    surface: Any = "browser",
    *,
    router: Any = None,
    agent: Any = None,
    embed_fn: Callable[[Sequence[str]], Any] | None = None,
) -> dict:
    """Select one proposed browser/computer action; never execute it.

    Candidates stay in the choice head, rather than the state, so each remains
    visible when the observation is long. Large sets are narrowed to 20 using
    the answering checkpoint's own embeddings before the decision pass.
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ToolError("invalid_goal", "goal must be a non-empty string")
    if not isinstance(observation, (str, dict)) or not observation:
        raise ToolError("invalid_observation", "observation must be non-empty text or a JSON object")
    if surface not in ("browser", "computer", "general"):
        raise ToolError("invalid_surface", "surface must be browser, computer or general")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ToolError("invalid_history", "history must be a JSON array")
    if not isinstance(actions, list) or len(actions) < 2:
        raise ToolError("invalid_actions", "actions must contain at least two candidates")

    by_id: dict[str, dict] = {}
    criteria: dict[str, str] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ToolError("invalid_actions", f"actions[{index}] must be a JSON object")
        action_id = action.get("id")
        description = action.get("description")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ToolError("invalid_actions", f"actions[{index}].id must be a non-empty string")
        if action_id in by_id:
            raise ToolError("invalid_actions", f"duplicate action id {action_id!r}")
        if not isinstance(description, str) or not description.strip():
            raise ToolError("invalid_actions", f"actions[{index}].description must be a non-empty string")
        by_id[action_id] = action
        criteria[action_id] = description

    state = {"goal": goal, "observation": observation, "recent_actions": history, "surface": surface}
    questions = {"next_action": {
        "type": "choice",
        "instructions": "Which available action best advances the goal, given the current observation and recent actions?",
        "criteria": criteria,
    }}
    if agent is not None:
        started = time.perf_counter()
        if len(actions) > 20:
            from laya.shortlist import embed_fn_from_agent, predict_shortlist

            if embed_fn is None:
                embed_fn = embed_fn_from_agent(agent)
            result = predict_shortlist(agent, state, questions, embed_fn, k=20)
        else:
            result = agent.system_one(state, questions)
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ToolError("internal_error", "action checkpoint returned no answers")
        prediction = {"answers": result["answers"], "routing": {
            "model": "action-checkpoint", "repo": None, "reason": "local action checkpoint",
        }, "latency_ms": round((time.perf_counter() - started) * 1000.0, 3)}
        if "shortlist" in result:
            prediction["shortlist"] = result["shortlist"]
        device = agent_device(agent)
        if device:
            prediction["device"] = device
    elif len(actions) > 20:
        prediction = laya_shortlist(state, questions, k=20, router=router, embed_fn=embed_fn)
    else:
        prediction = laya_predict(state, questions, router=router)

    answer = prediction["answers"].get("next_action")
    if not isinstance(answer, dict) or answer.get("choice") not in by_id:
        raise ToolError("internal_error", "checkpoint returned an unknown action id")
    chosen = answer["choice"]
    return {
        "action_id": chosen,
        "action": by_id[chosen],
        "confidence": answer.get("answer_confidence", answer.get("confidence")),
        "probabilities": answer.get("probabilities"),
        "candidate_count": len(actions),
        "routing": prediction["routing"],
        "latency_ms": prediction["latency_ms"],
        **({"device": prediction["device"]} if "device" in prediction else {}),
        **({"shortlist": prediction["shortlist"].get("next_action")}
           if "shortlist" in prediction else {}),
        "warning": ("Base checkpoints have not been validated for browser/computer action selection; "
                    "use a task-specific local checkpoint for reliable decisions.") if agent is None else None,
    }
