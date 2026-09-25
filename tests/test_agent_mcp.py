"""The standalone MCP extension works with the official Laya distribution."""

import asyncio
import json
import unittest
from unittest.mock import patch

from laya.mcp.server import server
from laya.mcp.tools import ToolError

from laya_tools import agent_mcp
from laya_tools.decisions import laya_choose_action
from laya_tools.shortlist import laya_shortlist


class FakeRouter:
    _agents = {}

    def predict(self, state, questions, **kwargs):
        self.state = state
        self.questions = questions
        return {
            "answers": {"next_action": {
                "choice": "click", "answer_confidence": 0.78,
                "probabilities": {"click": 0.9, "done": 0.1},
            }},
            "routing": {"model": "english", "reason": "test"},
        }


class ActionTests(unittest.TestCase):
    actions = [
        {"id": "click", "description": "Click the result", "target": "#result"},
        {"id": "done", "description": "Finish"},
    ]

    def test_choice_uses_candidate_ids_and_preserves_action(self):
        router = FakeRouter()
        result = laya_choose_action("Open result", "Result is visible", self.actions, router=router)
        self.assertEqual(result["action_id"], "click")
        self.assertEqual(result["action"], self.actions[0])
        self.assertEqual(result["confidence"], 0.78)
        self.assertEqual(router.questions["next_action"]["criteria"]["click"], "Click the result")
        self.assertEqual(router.state["goal"], "Open result")

    def test_invalid_actions(self):
        with self.assertRaises(ToolError):
            laya_choose_action("Open result", "Result visible", [self.actions[0]], router=FakeRouter())

    def test_mcp_wrapper(self):
        with patch.object(agent_mcp, "_router_or_error", return_value=FakeRouter()):
            result = json.loads(agent_mcp.laya_choose_action_tool(
                "Open result", "Result visible", self.actions))
        self.assertEqual(result, {"action_id": "click", "confidence": 0.78})

    def test_mcp_decision_tools_return_only_selected_values(self):
        full = json.dumps({"answers": {
            "kind": {"type": "choice", "choice": "click", "answer_confidence": 0.8,
                     "probabilities": {"click": 0.8, "done": 0.2}},
            "priority": {"type": "score", "score": 1.25, "confidence": 0.6,
                         "legend": {"0": "low", "1": "high"}},
            "safe": {"type": "noul", "noul": 0.7, "confidence": 0.7},
        }, "routing": {"model": "english"}, "latency_ms": 30, "device": "cpu"})
        expected = {"answers": {"kind": {"choice": "click", "confidence": 0.8},
                                 "priority": {"score": 1.25, "confidence": 0.6},
                                 "safe": {"noul": 0.7}}}
        with patch.object(agent_mcp, "upstream_predict_tool", return_value=full), \
             patch.object(agent_mcp, "upstream_preset_tool", return_value=full):
            self.assertEqual(json.loads(agent_mcp.laya_predict_tool({}, {})), expected)
            self.assertEqual(json.loads(agent_mcp.laya_preset_tool("triage", {})), expected)

    def test_mcp_route_and_status_are_compact(self):
        with patch.object(agent_mcp, "upstream_route_tool", return_value=json.dumps({
            "model": "english", "repo": "some/repo", "reason": "language"})), \
             patch.object(agent_mcp, "upstream_status_tool", return_value=json.dumps({
                 "device": "cpu", "device_is_preference": True, "torch_cuda": False,
                 "loaded": [], "checkpoint_devices": {}, "package_versions": {"torch": "x"}})):
            self.assertEqual(json.loads(agent_mcp.laya_route_tool({}, {})), {"model": "english"})
            self.assertEqual(json.loads(agent_mcp.laya_status_tool()), {
                "device": "cpu", "device_is_preference": True, "torch_cuda": False,
                "loaded": [], "checkpoint_devices": {}})

    def test_mcp_shortlist_returns_only_decision(self):
        raw = json.dumps({"answers": {"next": {"choice": "click", "answer_confidence": 0.9,
                                                   "probabilities": {"click": 0.9}}},
                          "shortlist": {"next": {"n": 50}}, "routing": {"model": "english"}})
        with patch.object(agent_mcp, "_router_or_error", return_value=FakeRouter()), \
             patch.object(agent_mcp, "_wrap", return_value=raw):
            result = json.loads(agent_mcp.laya_shortlist_tool({}, {}))
        self.assertEqual(result, {"answers": {"next": {"choice": "click", "confidence": 0.9}}})

    def test_tool_registration(self):
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        self.assertEqual(names, {"laya_status", "laya_route", "laya_predict", "laya_shortlist",
                                 "laya_preset", "laya_choose_action"})

    def test_shortlist_limits_choice_before_prediction(self):
        class Agent:
            def predict(self, state, questions):
                self.criteria = questions["next"]["criteria"]
                return {"answers": {"next": {"choice": next(iter(self.criteria))}}}

        agent = Agent()
        questions = {"next": {"type": "choice", "instructions": "Pick one", "criteria": {
            "a": "first", "b": "second", "c": "third"}}}
        embeddings = lambda texts: [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]
        result = laya_shortlist({"text": "test"}, questions, model="english", k=2,
                                agent=agent, embed_fn=embeddings)
        self.assertEqual(len(agent.criteria), 2)
        self.assertIn(result["answers"]["next"]["choice"], agent.criteria)
        self.assertEqual(result["shortlist"]["next"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
