"""TUI request building and UI flow without downloading model weights."""

import json
import unittest
from unittest.mock import patch

from textual.widgets import Input, Select, TextArea
from textual.events import MouseDown, MouseMove, MouseUp
from laya_tools.tui import LayaTUI, ResizeHandle, answer_summary, prepare_request, run_request, write_system_clipboard


class RequestTests(unittest.TestCase):
    def test_score_summary_shows_rubric_and_every_answer_field(self):
        questions = {
            "difficulty": {"type": "score", "instructions": "How hard is this?",
                           "criteria": ["trivial", "moderate", "hard"]},
            "domain": {"type": "choice", "instructions": "Which domain?",
                       "criteria": {"code": "programming", "math": "calculation"}},
            "needs_tools": {"type": "noul", "instructions": "Are tools needed?",
                            "criteria": {"false": "local answer", "true": "external lookup"}},
        }
        result = {"answers": {
            "difficulty": {"type": "score", "score": 1.25,
                           "legend": {"0": "trivial", "1": "moderate", "2": "hard"},
                           "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
                           "confidence": 0.32, "answer_confidence": 0.55,
                           "action": {"act_probability": 0.9}},
            "domain": {"type": "choice", "choice": "code",
                       "probabilities": {"code": 0.8, "math": 0.2},
                       "confidence": 0.7, "answer_confidence": 0.8},
            "needs_tools": {"type": "noul", "noul": 0.75,
                            "confidence": 0.75, "answer_confidence": 0.75},
        }}
        summary = answer_summary(result, questions)
        for expected in ("score=1.25", "0 (trivial)=0.1000",
                         "1 (moderate)=0.5500", "2 (hard)=0.3500",
                         "answer_confidence=0.55", "act_probability", "programming",
                         "calculation", "P(true)=0.7500", "P(false)=0.2500",
                         "external lookup"):
            self.assertIn(expected, summary)

    def test_score_summary_uses_question_criteria_without_legend(self):
        result = {"answers": {"severity": {"type": "score", "score": 0.4,
                                          "probabilities": {"0": 0.6, "1": 0.4},
                                          "extra_metric": "kept"}}}
        questions = {"severity": {"type": "score", "criteria": ["low", "high"]}}
        summary = answer_summary(result, questions)
        self.assertIn("0 (low)=0.6000", summary)
        self.assertIn("1 (high)=0.4000", summary)
        self.assertIn("extra_metric=kept", summary)

    def test_desktop_clipboard_writer(self):
        with patch("laya_tools.tui.clipboard_commands", return_value=[("xclip", "-selection", "clipboard")]), \
             patch("laya_tools.tui.shutil.which", return_value="/usr/bin/xclip"), \
             patch("laya_tools.tui.subprocess.run") as run:
            self.assertTrue(write_system_clipboard("copied text"))
        self.assertEqual(run.call_args.args[0], ("/usr/bin/xclip", "-selection", "clipboard"))
        self.assertEqual(run.call_args.kwargs["input"], "copied text")

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
    async def test_result_view_keeps_question_context_and_metadata(self):
        app = LayaTUI()
        async with app.run_test(size=(100, 30)):
            app.finish_request(
                {"routing": {"model": "english", "reason": "test", "repo": "example"},
                 "answers": {"urgency": {"type": "score", "score": 0.8,
                                         "probabilities": {"0": 0.2, "1": 0.8}}},
                 "usage": {"input_tokens": 12}, "device": "cpu"},
                None,
                {"urgency": {"type": "score", "instructions": "How urgent?",
                             "criteria": ["no rush", "urgent"]}},
            )
            text = app.query_one("#results", TextArea).text
            for expected in ("reason=test", "repo=example", "How urgent?",
                             "1 (urgent)=0.8000", "input_tokens", "device=cpu"):
                self.assertIn(expected, text)

    @patch("laya_tools.tui.write_system_clipboard", return_value=True)
    async def test_copy_paste_and_mouse_resize(self, system_copy):
        app = LayaTUI()
        async with app.run_test(size=(100, 38)) as pilot:
            prompt = app.query_one("#prompt", TextArea)
            questions = app.query_one("#questions", TextArea)
            results = app.query_one("#results", TextArea)
            self.assertTrue(results.read_only)

            prompt.text = "A prompt to copy"
            await pilot.click("#copy-prompt")
            self.assertEqual(app.clipboard, prompt.text)
            system_copy.assert_called_with(prompt.text)
            questions.text = '{"question": "copy me"}'
            await pilot.click("#copy-questions")
            self.assertEqual(app.clipboard, questions.text)
            app.copy_to_clipboard(" pasted")
            prompt.focus()
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.press("ctrl+v")
            self.assertEqual(prompt.text, "A prompt to copy pasted")

            app.history = [{"answers": {"decision": {"choice": "yes"}}}]
            app.show_history()
            await pilot.click("#copy-results")
            self.assertEqual(app.clipboard, results.text)
            system_copy.assert_called_with(results.text)
            results.focus()
            results.action_select_all()
            results.action_copy()
            self.assertEqual(app.clipboard, results.text)

            handle = app.query_one("#resize-prompt", ResizeHandle)
            original_height = prompt.outer_size.height
            def mouse(event_type, y):
                return event_type(handle, 0, 0, 0, 0, 1, False, False, False, screen_y=y)
            handle.on_mouse_down(mouse(MouseDown, 10))
            handle.on_mouse_move(mouse(MouseMove, 12))
            await pilot.pause()
            self.assertEqual(prompt.outer_size.height, original_height + 2)
            handle.on_mouse_up(mouse(MouseUp, 12))

            handle = app.query_one("#resize-questions", ResizeHandle)
            original_questions = questions.outer_size.height
            original_results = results.outer_size.height
            handle.on_mouse_down(mouse(MouseDown, 20))
            handle.on_mouse_move(mouse(MouseMove, 22))
            await pilot.pause()
            self.assertEqual(questions.outer_size.height, original_questions + 2)
            self.assertEqual(results.outer_size.height, original_results - 2)
            handle.on_mouse_up(mouse(MouseUp, 22))

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
            self.assertGreaterEqual(app.query_one("#results", TextArea).outer_size.height, 3)
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
