# LayaTUI-MCP

A terminal UI and an MCP server for [Laya](https://pypi.org/project/laya/). This project installs `laya==0.3.20` from PyPI; it does not require a checkout of Laya's upstream repository.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and run these commands from its root:

```bash
uv sync --frozen
uv run laya-tui
```

`uv` creates and manages the virtual environment. After the first sync, `uv run laya-tui` is enough. To launch the MCP server manually:

```bash
uv run laya-agent-mcp
```

The MCP server uses stdio. An MCP client starts it as a subprocess; running it directly will not display a UI or a response. Laya model weights are downloaded the first time each model is used.

The TUI accepts plain text or JSON state, supports built-in or custom questions, shows routing and answers, and can save the latest JSON result. Select text in **Prompt**, **Questions**, or **Result** and press `Ctrl+C` to copy it. Press `Ctrl+V` in Prompt or Questions to paste. `F7` selects the entire focused field, and each field has a button to copy all of its text. Result is read-only. Drag the separators below Prompt and Questions to resize those fields and the remaining Result area with the mouse.

Copy writes to the desktop clipboard through `xclip` or `xsel` on X11, `wl-copy` on Wayland, and `pbcopy` on macOS. Install one of those utilities if needed. Textual's terminal clipboard sequence is also attempted; if no desktop clipboard writer succeeds, the TUI shows a warning instead of claiming the copy succeeded.

The device selector supports CPU, CUDA, MPS, XPU, and a custom identifier such as `cuda:1`.

For a `score` question, `criteria` is an **ordered list** of rubric levels. Index `0` is the first level. The result displays the probability of every level next to its description and reports `score` as the probability-weighted expected level, which may be fractional. In Summary, instructions, the selected or most likely option, and general information use distinct colors. The Summary view also includes routing details, confidence values, action metadata, token usage, and response time in milliseconds. The timer includes model loading on first use. Raw JSON and saved results include `response_time_ms`.

The MCP server exposes `laya_status`, `laya_route`, `laya_predict`, `laya_preset`, `laya_shortlist`, and `laya_choose_action`, plus the `laya_agent_decisions` prompt. Its responses are compact: `laya_route` returns the model name; prediction tools return each selected choice, score, or true probability with confidence where useful; `laya_choose_action` returns only `action_id` and `confidence`. The agent can match that ID to the candidate action it supplied. Routing details, full probability distributions, latency, and shortlist metadata stay out of MCP responses. `laya_status` reports the device and loaded checkpoints.

`laya_choose_action` accepts a goal, current observation, and candidate actions; it returns a choice without executing it. With more than 20 actions, it uses embeddings to shortlist them first. For reliable navigation decisions, set `LAYA_ACTION_MODEL` to a local checkpoint trained for the `next_action` schema. The general checkpoints have not been validated for this task.

Set `LAYA_DEVICE=cpu` or `LAYA_DEVICE=cuda` to choose the MCP compute device. The server loads models on first use by default (`LAYA_PRELOAD=0`).

## Connect the MCP server to an agent

Run `uv sync --frozen` from the repository root before configuring a client. In the examples below, replace `/ABSOLUTE/PATH/LayaTUI-MCP` with the output of `pwd` and `/ABSOLUTE/PATH/uv` with the output of `command -v uv`. Clients can start the MCP server from any directory: `uv --directory` selects this project. `--frozen` uses the included `uv.lock`.

### Claude Code

```bash
REPO="$(pwd)"
UV="$(command -v uv)"
claude mcp add --scope user laya -- "$UV" --directory "$REPO" run --frozen laya-agent-mcp
claude mcp list
```

User scope makes the server available across projects. Use `--scope project` for project scope. [Claude Code documentation](https://code.claude.com/docs/en/mcp).

### Codex

```bash
REPO="$(pwd)"
UV="$(command -v uv)"
codex mcp add laya -- "$UV" --directory "$REPO" run --frozen laya-agent-mcp
codex mcp list
```

Codex stores the connection in `~/.codex/config.toml`. [Official OpenAI documentation](https://developers.openai.com/codex/mcp).

### OpenCode

Add the `laya` entry to the existing `mcp` object in `~/.config/opencode/opencode.json`. For classic OpenCode:

```json
{
  "mcp": {
    "laya": {
      "type": "local",
      "command": ["/ABSOLUTE/PATH/uv", "--directory", "/ABSOLUTE/PATH/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"],
      "enabled": true
    }
  }
}
```

In OpenCode V2, place the same entry under `mcp.servers` and omit `enabled`. Verify with `opencode mcp list` (or `opencode2 mcp list` for V2). [OpenCode documentation](https://opencode.ai/docs/mcp-servers/) and [OpenCode V2 documentation](https://dev.opencode.ai/v2/docs/mcp-servers/).

### Hermes Agent

Add the entry under `mcp_servers` in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  laya:
    command: /ABSOLUTE/PATH/uv
    args: ["--directory", "/ABSOLUTE/PATH/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"]
    connect_timeout: 60
```

Verify with `hermes mcp test laya`. [Hermes Agent reference](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference).

### LM Studio and other `mcp.json` clients

In LM Studio, open **Program → Install → Edit mcp.json** and add `laya` to the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "laya": {
      "command": "/ABSOLUTE/PATH/uv",
      "args": ["--directory", "/ABSOLUTE/PATH/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"]
    }
  }
}
```

This format is also a starting point for Cursor and other clients that use `mcpServers`. [LM Studio documentation](https://lmstudio.ai/docs/app/mcp).

To encourage an agent to consult Laya when choosing among visible actions, give it an instruction such as: “When several concrete browser or computer actions are available, inspect the screen first and call `laya_choose_action` with the goal, observation, and actions with unique IDs. Execute the chosen action with your normal browser or computer tool.” Laya makes the choice; it does not navigate or inspect image pixels itself.
