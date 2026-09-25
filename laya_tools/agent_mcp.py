"""Extend the PyPI Laya MCP server with a browser/computer action decision tool."""

import os
import threading
from pathlib import Path

from laya.mcp.server import _router_or_error, _wrap, main as laya_mcp_main, server
from laya.mcp.tools import ToolError

from .decisions import laya_choose_action
from .shortlist import laya_shortlist

_ACTION_AGENT = None
_ACTION_AGENT_LOCK = threading.Lock()


@server.tool(
    name="laya_shortlist",
    description="Shortlist large choice questions using Laya embeddings, then predict among the remaining labels.",
)
def laya_shortlist_tool(state: dict, questions: dict, model: str = "auto", k: int = 20) -> str:
    """Shortlist choice labels before making a typed decision."""
    return _wrap(laya_shortlist, state=state, questions=questions, model=model, k=k,
                 router=_router_or_error())


def _ensure_action_agent():
    global _ACTION_AGENT
    if _ACTION_AGENT is not None:
        return _ACTION_AGENT
    with _ACTION_AGENT_LOCK:
        if _ACTION_AGENT is not None:
            return _ACTION_AGENT
        path = Path(os.environ["LAYA_ACTION_MODEL"]).expanduser()
        if not path.is_dir():
            raise ToolError("models_not_ready", f"LAYA_ACTION_MODEL is not a local checkpoint directory: {path}")
        try:
            from laya import load

            agent = load(str(path), device=os.environ.get("LAYA_DEVICE") or None)
            cfg = getattr(agent, "cfg", None)
            if isinstance(cfg, dict) and cfg.get("head_max_len_train"):
                cfg["head_max_len"] = cfg["head_max_len_train"]
        except Exception as exc:
            raise ToolError("models_not_ready", f"action checkpoint could not be loaded: {exc}") from exc
        _ACTION_AGENT = agent
        return agent


def _run_with_action_agent(**kwargs):
    return laya_choose_action(**kwargs, agent=_ensure_action_agent())


@server.tool(
    name="laya_choose_action",
    description=(
        "Use when a browser-use or computer-use agent must choose its next action from concrete "
        "candidates after inspecting the current page or screen. Pass a goal, textual observation "
        "(DOM/accessibility tree/OCR or screenshot summary), recent history, and candidate actions "
        "with unique ids and descriptions. Include a DONE or wait action when applicable. "
        "Returns the selected id, original action, confidence, probabilities and routing; it never "
        "executes the action. More than 20 candidates are automatically shortlisted. "
        "For reliable browser action selection set LAYA_ACTION_MODEL to a locally fine-tuned "
        "checkpoint directory compatible with this tool's next_action schema; base checkpoints "
        "are unvalidated for this task. Do NOT use for open-ended generation."
    ),
)
def laya_choose_action_tool(goal: str, observation: str | dict, actions: list[dict],
                            history: list | None = None, surface: str = "browser") -> str:
    """Choose one browser/computer action without executing it."""
    kwargs = {"goal": goal, "observation": observation, "actions": actions,
              "history": history, "surface": surface}
    if os.environ.get("LAYA_ACTION_MODEL", "").strip():
        return _wrap(_run_with_action_agent, **kwargs)
    return _wrap(laya_choose_action, **kwargs, router=_router_or_error())


@server.prompt(
    name="laya_agent_decisions",
    description="Instructions for using Laya when an agent chooses among browser or computer actions.",
)
def laya_agent_decisions() -> str:
    return (
        "Use laya_choose_action whenever you must select the next browser or computer action "
        "from concrete visible candidates. First inspect the current page or screen with your "
        "normal browser/computer tool. Provide the goal, a concise textual observation, recent "
        "actions, and candidate actions with unique IDs and descriptions that identify the "
        "operation and target. Include DONE when completion is possible. Review the returned "
        "confidence and probabilities, and use your normal browser/computer tool to execute "
        "the selected action only when the evidence supports it. Laya never executes actions, "
        "cannot inspect image pixels itself, and does not generate free-form answers."
    )


def main():
    os.environ.setdefault("LAYA_PRELOAD", "0")
    laya_mcp_main()


if __name__ == "__main__":
    main()
