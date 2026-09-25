"""TUI request building and UI flow without downloading model weights."""

import json
import unittest

from textual.widgets import Input, Select, TextArea
from laya_tools.tui import LayaTUI, prepare_request, run_request


class RequestTests(unittest.TestCase):
    def test_text_preset_and_options(self):
        questions = {"answer": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}}
        state, actual_questions, options = prepare_request(
            "hello", json.dumps(questions), "triage", "text", "multilingual", "", "es", "8192", "32")
        self.assertEqual(state, {"message": "hello"})
        self.assertEqual(actual_questions, questions)
        self.assertEqual(options["model"], "multilingual")
        self.assertEqual(options["max_len"], 8192)
        self.assertEqual(options["head_max_len"], 32)

    def test_structured_state_and_validation(self):
        questions = json.dumps({"safe": {"type": "noul", "instructions": "Is it safe?"}})
        state, _, _ = prepare_request('{"body":"hola"}', questions, "custom", "json", "", "", "", "", "")
        self.assertEqual(state, {"body": "hola"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            prepare_request("hello", questions, "custom", "text", "", "", "", "-1", "")
        with self.assertRaisesRegex(ValueError, "State is not valid JSON"):
            prepare_request("{", questions, "custom", "json", "", "", "", "", "")

    def test_route_omits_prediction_only_options(self):
        class Router:
            def route(self, state, questions, **options):
                self.options = options
                return {"model": "english", "reason": "test"}

        router = Router()
        result = run_request(router, "route", {"text": "hi"}, {},
                             {"model": None, "task": None, "lang": None, "max_len": 4096})
        self.assertEqual(result["routing"]["model"], "english")
        self.assertNotIn("max_len", router.options)


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_gpu_and_result_flow(self):
        instances = []

        class Router:
            def __init__(self, device=None):
                self.device = device
                instances.append(self)

            def predict(self, state, questions, **options):
                self.state = state
                return {"routing": {"model": "english", "reason": "test"},
                        "answers": {"answer": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 2}}

        app = LayaTUI(router_factory=Router)
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#prompt", TextArea).text = "hello"
            app.query_one("#device", Select).value = "cpu"
            await pilot.click("#send")
            await pilot.pause()
            self.assertEqual(instances[-1].device, "cpu")
            self.assertEqual(len(app.history), 1)
            app.query_one("#custom-device", Input).value = "cuda:1"
            await pilot.click("#send")
            await pilot.pause()
            self.assertEqual(instances[-1].device, "cuda:1")
            self.assertEqual(len(app.history), 2)
            app.query_one("#view", Select).value = "json"
            await pilot.pause()
            self.assertIn("Done", str(app.query_one("#status").render()))


if __name__ == "__main__":
    unittest.main()
