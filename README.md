# LayaTUI-MCP

TUI y servidor MCP para [Laya](https://pypi.org/project/laya/). El proyecto instala
`laya==0.3.20` desde PyPI; no necesita clonar el repositorio de Laya.

## Inicio rápido

Instalá [uv](https://docs.astral.sh/uv/getting-started/installation/), cloná este
repositorio y ejecutá desde su raíz:

```bash
uv sync --frozen
uv run laya-tui
```

`uv` crea y administra el entorno virtual automáticamente. Después del primer
`uv sync`, alcanza con `uv run laya-tui`. Para iniciar el MCP manualmente:

```bash
uv run laya-agent-mcp
```

El MCP habla por stdio: cuando lo usa un cliente, ese cliente inicia el proceso.
No imprime una interfaz ni una respuesta en la terminal. Los pesos de Laya se
descargan al primer uso de cada modelo.

La TUI permite enviar texto o estado JSON, elegir preguntas predefinidas o
personalizadas, inspeccionar rutas, ver respuestas resumidas o JSON y guardar
el último resultado. Su selector de dispositivo admite CPU, CUDA, MPS, XPU y
un identificador personalizado como `cuda:1`.

El MCP ofrece `laya_status`, `laya_route`, `laya_predict`, `laya_preset`,
`laya_shortlist` y `laya_choose_action`, además del prompt
`laya_agent_decisions`. `laya_choose_action` recibe un objetivo, la observación
actual y acciones candidatas; devuelve la acción elegida sin ejecutarla. Si
hay más de 20 acciones, primero reduce las opciones con embeddings. Para
decisiones de navegación confiables, configurá `LAYA_ACTION_MODEL` con un
checkpoint local entrenado para el esquema `next_action`; los checkpoints
generales todavía no están validados para esa tarea.

El servidor usa `LAYA_DEVICE=cpu` o `LAYA_DEVICE=cuda` para elegir dispositivo.
Carga los modelos al primer uso por defecto (`LAYA_PRELOAD=0`).

## Conectar el MCP a un agente

Primero ejecutá `uv sync --frozen` desde la raíz del repositorio. En los
ejemplos, reemplazá `/RUTA/ABSOLUTA/LayaTUI-MCP` por el resultado de `pwd` y
`/RUTA/ABSOLUTA/uv` por el resultado de `command -v uv`. Los clientes MCP
pueden iniciarse desde cualquier directorio: `uv --directory` selecciona este
proyecto. Usamos `--frozen` para respetar el `uv.lock` incluido.

### Claude Code

```bash
REPO="$(pwd)"
UV="$(command -v uv)"
claude mcp add --scope user laya -- "$UV" --directory "$REPO" run --frozen laya-agent-mcp
claude mcp list
```

`--scope user` lo deja disponible en todos los proyectos. Para limitarlo al
proyecto actual, cambiá `user` por `project`. [Documentación de Claude
Code](https://code.claude.com/docs/en/mcp).

### Codex

```bash
REPO="$(pwd)"
UV="$(command -v uv)"
codex mcp add laya -- "$UV" --directory "$REPO" run --frozen laya-agent-mcp
codex mcp list
```

Codex guarda la conexión en `~/.codex/config.toml`. [Documentación oficial de
OpenAI](https://developers.openai.com/codex/mcp).

### OpenCode

En `~/.config/opencode/opencode.json`, agregá la entrada `laya` al objeto `mcp`
existente. Para OpenCode clásico:

```json
{
  "mcp": {
    "laya": {
      "type": "local",
      "command": ["/RUTA/ABSOLUTA/uv", "--directory", "/RUTA/ABSOLUTA/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"],
      "enabled": true
    }
  }
}
```

En OpenCode V2, la misma entrada va dentro de `mcp.servers` y se omite
`enabled`. Verificá con `opencode mcp list` (o `opencode2 mcp list` en V2).
[Documentación de OpenCode](https://opencode.ai/docs/mcp-servers/) y
[OpenCode V2](https://dev.opencode.ai/v2/docs/mcp-servers/).

### Hermes Agent

En `~/.hermes/config.yaml`, agregá la entrada a `mcp_servers`:

```yaml
mcp_servers:
  laya:
    command: /RUTA/ABSOLUTA/uv
    args: ["--directory", "/RUTA/ABSOLUTA/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"]
    connect_timeout: 60
```

Verificá con `hermes mcp test laya`. [Referencia de Hermes
Agent](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference).

### LM Studio y clientes compatibles con `mcp.json`

En LM Studio, abrí **Program → Install → Edit mcp.json** y agregá `laya` al
objeto `mcpServers` existente:

```json
{
  "mcpServers": {
    "laya": {
      "command": "/RUTA/ABSOLUTA/uv",
      "args": ["--directory", "/RUTA/ABSOLUTA/LayaTUI-MCP", "run", "--frozen", "laya-agent-mcp"]
    }
  }
}
```

Este formato también sirve como base para Cursor y otros clientes que usan
`mcpServers`. [Documentación de LM Studio](https://lmstudio.ai/docs/app/mcp).

Para que el modelo consulte Laya al decidir entre acciones visibles, podés
darle esta instrucción en tu agente: “Cuando tengas varias acciones concretas
de navegación o uso de computadora, inspeccioná primero la pantalla y llamá a
`laya_choose_action` con el objetivo, la observación y las acciones con IDs
únicos. Ejecutá la acción elegida con tu herramienta habitual”. Laya toma la
decisión; no navega ni ve píxeles por sí mismo.
