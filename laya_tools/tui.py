"""Terminal interface for Laya's local routing and typed predictions."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, TextArea

from laya import DEFAULT_MODELS, Router
from laya.cli import PRESETS

PRESET_STATE_KEYS = {
    "email": "body",
    "guard": "prompt",
    "moderation": "post",
    "router": "request",
    "triage": "message",
}


def clipboard_commands():
    """Available desktop clipboard writers, in session-preferred order."""
    if sys.platform == "darwin":
        return [("pbcopy",)]
    if os.name == "nt":
        return [("clip.exe",)]
    if os.environ.get("WAYLAND_DISPLAY"):
        return [("wl-copy",), ("xclip", "-selection", "clipboard"), ("xsel", "--clipboard", "--input")]
    if os.environ.get("DISPLAY"):
        return [("xclip", "-selection", "clipboard"), ("xsel", "--clipboard", "--input"), ("wl-copy",)]
    return []


def write_system_clipboard(text: str) -> bool:
    """Write to the OS clipboard without invoking a shell or exposing the text in argv."""
    for command in clipboard_commands():
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            subprocess.run((executable, *command[1:]), input=text, text=True,
                           encoding="utf-8", stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3, check=True)
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def prepare_request(prompt, questions_text, preset, state_format, model, task, lang,
                    max_len, head_max_len):
    """Validate the form and build arguments for Router.route / Router.predict."""
    if not prompt.strip():
        raise ValueError("Enter a prompt or load a file first.")
    if state_format == "json":
        try:
            state = json.loads(prompt)
        except json.JSONDecodeError as error:
            raise ValueError("State is not valid JSON: %s" % error) from error
        if not isinstance(state, (str, dict, list)):
            raise ValueError("JSON state must be a string, object or array.")
    else:
        state = {PRESET_STATE_KEYS.get(preset, "text"): prompt}

    try:
        questions = json.loads(questions_text)
    except json.JSONDecodeError as error:
        raise ValueError("Questions are not valid JSON: %s" % error) from error
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a nonempty JSON object.")
    for name, question in questions.items():
        if not isinstance(name, str) or not isinstance(question, dict):
            raise ValueError("Each question must have a string ID and an object definition.")
        if question.get("type") not in ("choice", "score", "noul"):
            raise ValueError("Question %r needs type choice, score or noul." % name)
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError("Question %r needs instructions." % name)
        criteria = question.get("criteria")
        if question["type"] == "choice" and (not isinstance(criteria, (dict, list)) or not criteria):
            raise ValueError("Choice question %r needs nonempty criteria (object or array)." % name)
        if question["type"] == "score" and (not isinstance(criteria, list) or not criteria):
            raise ValueError("Score question %r needs a nonempty criteria array." % name)

    def positive_int(value, label):
        if not value.strip():
            return None
        try:
            number = int(value)
        except ValueError as error:
            raise ValueError("%s must be a positive integer." % label) from error
        if number <= 0:
            raise ValueError("%s must be a positive integer." % label)
        return number

    options = {"model": model or None, "task": task.strip() or None, "lang": lang.strip() or None}
    if max_len.strip():
        options["max_len"] = positive_int(max_len, "max_len")
    if head_max_len.strip():
        options["head_max_len"] = positive_int(head_max_len, "head_max_len")
    return state, questions, options


def run_request(router, mode, state, questions, options):
    """Run a prepared request; route mode does not load model weights."""
    if mode == "route":
        routing = router.route(state, questions, **{key: options[key] for key in ("model", "task", "lang")})
        return {"routing": dict(routing)}
    return router.predict(state, questions, **options)


def answer_summary(result):
    """A plain-text summary that can be selected and copied in the result field."""
    rows = []
    for name, answer in result.get("answers", {}).items():
        if "choice" in answer:
            choice = str(answer["choice"])
            probability = answer.get("probabilities", {}).get(choice)
            detail = "p=%.3f" % probability if isinstance(probability, (float, int)) else ""
            rows.append(f"{name}: {choice}" + (f" ({detail})" if detail else ""))
        elif "score" in answer:
            rows.append(f"{name}: {answer['score']} (score)")
        elif "noul" in answer:
            rows.append(f"{name}: {answer['noul']:.3f} (probability of yes)")
        else:
            rows.append(f"{name}: {json.dumps(answer, ensure_ascii=False, default=str)}")
    return "\n".join(rows)


class ResizeHandle(Static):
    """Drag to change a text field's height while keeping room for results."""

    def __init__(self, target_id: str, **kwargs):
        super().__init__("↕ Drag to resize", **kwargs)
        self.target_id = target_id
        self._drag_start = None

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        target = self.app.query_one(f"#{self.target_id}", TextArea)
        results = self.app.query_one("#results", TextArea)
        self._drag_start = (event.screen_y, target.outer_size.height, results.outer_size.height)
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: MouseMove) -> None:
        if self._drag_start is None:
            return
        start_y, start_height, results_height = self._drag_start
        delta = round(event.screen_y - start_y)
        height = max(3, min(start_height + delta, start_height + max(0, results_height - 3)))
        self.app.query_one(f"#{self.target_id}", TextArea).styles.height = height
        event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._drag_start is not None:
            self._drag_start = None
            self.release_mouse()
            event.stop()


class LayaTUI(App):
    """Interactive frontend for local Laya decisions."""

    TITLE = "Laya"
    SUB_TITLE = "Local decision engine"
    BINDINGS = [("ctrl+enter", "send", "Send"), ("ctrl+l", "clear", "Clear results"), ("ctrl+q", "quit", "Quit")]
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #settings { width: 38; min-width: 32; border: round $primary; padding: 0 1; overflow-y: auto; }
    #main { width: 1fr; padding: 0 1; }
    .caption { height: 1; margin-top: 1; color: $text-muted; }
    #prompt { height: 5; border: round $primary; }
    #questions { height: 6; border: round $primary; }
    #results { height: 1fr; min-height: 3; border: round $primary; }
    ResizeHandle { height: 1; text-align: center; color: $text-muted; pointer: ns-resize; }
    #status { height: 1; color: $accent; }
    #buttons { height: 3; }
    #buttons Button { margin-right: 1; }
    #file-buttons { height: 3; }
    #file-buttons Button { margin-right: 1; }
    #main Button.copy { min-width: 17; height: 1; min-height: 1; border: none; padding: 0 1; margin-left: 1; }
    #main .field-heading { height: 1; align: left middle; }
    Input, Select { margin-bottom: 1; }
    """

    def __init__(self, router_factory=Router):
        super().__init__()
        self.router_factory = router_factory
        self.router = None
        self.busy = False
        self.history = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with VerticalScroll(id="settings"):
                yield Label("Mode", classes="caption")
                yield Select([("Predict", "predict"), ("Route only", "route")], value="predict", id="mode")
                yield Label("Question set", classes="caption")
                yield Select([("Custom", "custom")] + [(name.title(), name) for name in PRESETS],
                             value="router", id="preset")
                yield Label("Checkpoint", classes="caption")
                yield Select([("Auto", "")] + [(name, name) for name in DEFAULT_MODELS],
                             value="", id="model")
                yield Label("State format", classes="caption")
                yield Select([("Prompt text", "text"), ("JSON state", "json")], value="text", id="state-format")
                yield Input(placeholder="Language override (e.g. es)", id="lang")
                yield Input(placeholder="Task override (e.g. typed_decisions)", id="task")
                yield Label("Compute device", classes="caption")
                yield Select([("Auto", ""), ("CPU", "cpu"), ("NVIDIA GPU (CUDA)", "cuda"),
                              ("Apple GPU (MPS)", "mps"), ("Intel GPU (XPU)", "xpu")],
                             value="", id="device")
                yield Input(placeholder="Custom device (e.g. cuda:1)", id="custom-device")
                yield Input(placeholder="max_len (blank = default)", id="max-len")
                yield Input(placeholder="head_max_len (blank = default)", id="head-max-len")
                yield Label("File path: load prompt / save latest JSON", classes="caption")
                yield Input(placeholder="Path", id="path")
                with Horizontal(id="file-buttons"):
                    yield Button("Load", id="load")
                    yield Button("Save result", id="save")
                yield Label("Result view", classes="caption")
                yield Select([("Summary", "summary"), ("Raw JSON", "json")], value="summary", id="view")
            with Vertical(id="main"):
                with Horizontal(classes="field-heading"):
                    yield Label("Prompt or JSON state")
                    yield Button("Copy prompt", id="copy-prompt", classes="copy")
                yield TextArea(id="prompt")
                yield ResizeHandle("prompt", id="resize-prompt")
                with Horizontal(classes="field-heading"):
                    yield Label("Questions JSON (choice, score, noul)")
                    yield Button("Copy questions", id="copy-questions", classes="copy")
                yield TextArea(json.dumps(PRESETS["router"](), ensure_ascii=False, indent=2),
                               language="json", id="questions")
                yield ResizeHandle("questions", id="resize-questions")
                with Horizontal(id="buttons"):
                    yield Button("Send  Ctrl+Enter", variant="primary", id="send")
                    yield Button("Clear results", id="clear")
                yield Label("Ready", id="status")
                with Horizontal(classes="field-heading"):
                    yield Label("Result (select text to copy)")
                    yield Button("Copy result", id="copy-results", classes="copy")
                yield TextArea(id="results", read_only=True, show_line_numbers=False)
        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "preset" and event.value != "custom":
            self.query_one("#questions", TextArea).text = json.dumps(
                PRESETS[event.value](), ensure_ascii=False, indent=2)
        elif event.select.id == "view":
            self.show_history()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {"send": self.action_send, "clear": self.action_clear,
                   "load": self.action_load, "save": self.action_save}
        action = actions.get(event.button.id)
        if action:
            action()
        elif event.button.id and event.button.id.startswith("copy-"):
            self.copy_field(event.button.id.removeprefix("copy-"))

    def copy_field(self, field_id: str) -> None:
        text = self.query_one(f"#{field_id}", TextArea).text
        if text:
            if self.copy_to_clipboard(text):
                self.set_status(f"Copied {field_id} to system clipboard.")
            else:
                self.set_status("System clipboard unavailable; terminal copy was attempted.")
        else:
            self.set_status(f"{field_id.title()} is empty.")

    def copy_to_clipboard(self, text: str) -> bool:
        """Keep Textual's local/OSC 52 copy and also write the desktop clipboard."""
        super().copy_to_clipboard(text)
        copied = write_system_clipboard(text)
        if not copied:
            self.notify("System clipboard unavailable. Check terminal clipboard support.",
                        severity="warning", timeout=5)
        return copied

    def set_status(self, message):
        self.query_one("#status", Label).update(message)

    def action_send(self) -> None:
        if self.busy:
            self.set_status("A request is already running.")
            return
        try:
            mode = self.query_one("#mode", Select).value
            device = (self.query_one("#custom-device", Input).value.strip()
                      or self.query_one("#device", Select).value or None)
            state, questions, options = prepare_request(
                self.query_one("#prompt", TextArea).text,
                self.query_one("#questions", TextArea).text,
                self.query_one("#preset", Select).value,
                self.query_one("#state-format", Select).value,
                self.query_one("#model", Select).value,
                self.query_one("#task", Input).value,
                self.query_one("#lang", Input).value,
                self.query_one("#max-len", Input).value,
                self.query_one("#head-max-len", Input).value,
            )
        except ValueError as error:
            self.set_status(str(error))
            return
        self.busy = True
        self.query_one("#send", Button).disabled = True
        self.set_status("Routing..." if mode == "route" else "Loading/running checkpoint (first use may download it)...")
        self.execute_request(mode, device, state, questions, options)

    @work(thread=True)
    def execute_request(self, mode, device, state, questions, options) -> None:
        try:
            if self.router is None or getattr(self, "_router_device", None) != device:
                router = self.router_factory(device=device)
                self.router = router
                self._router_device = device
            result = run_request(self.router, mode, state, questions, options)
        except Exception as error:
            self.call_from_thread(self.finish_request, None, str(error))
        else:
            self.call_from_thread(self.finish_request, result, None)

    def finish_request(self, result, error):
        self.busy = False
        self.query_one("#send", Button).disabled = False
        if error:
            self.set_status("Error: %s" % error)
            return
        self.history.append(result)
        self.show_history()
        self.set_status("Done. %d result(s)." % len(self.history))

    def show_history(self):
        lines = []
        raw = self.query_one("#view", Select).value == "json"
        for index, result in enumerate(self.history, 1):
            lines.append("Result %d" % index)
            if raw:
                lines.append(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                lines.append("")
                continue
            routing = result.get("routing", {})
            if routing:
                lines.append("Model: %s | %s" % (routing.get("model", "?"), routing.get("reason", "")))
                if routing.get("detection"):
                    lines.append("Detection: %s" % json.dumps(routing["detection"], ensure_ascii=False, default=str))
            if result.get("answers"):
                lines.append(answer_summary(result))
            if result.get("usage"):
                lines.append("Usage: %s" % json.dumps(result["usage"], ensure_ascii=False, default=str))
            lines.append("")
        self.query_one("#results", TextArea).text = "\n".join(lines).rstrip()

    def action_clear(self):
        self.history.clear()
        self.show_history()
        self.set_status("Results cleared.")

    def action_load(self):
        path = self.query_one("#path", Input).value.strip()
        if not path:
            self.set_status("Enter a file path first.")
            return
        try:
            self.query_one("#prompt", TextArea).text = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as error:
            self.set_status("Could not load: %s" % error)
        else:
            self.set_status("Loaded %s" % path)

    def action_save(self):
        path = self.query_one("#path", Input).value.strip()
        if not path or not self.history:
            self.set_status("Enter a file path and run a request first.")
            return
        try:
            Path(path).expanduser().write_text(
                json.dumps(self.history[-1], ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as error:
            self.set_status("Could not save: %s" % error)
        else:
            self.set_status("Saved latest result to %s" % path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive terminal interface for Laya")
    parser.parse_args(argv)
    LayaTUI().run()


if __name__ == "__main__":
    main()
