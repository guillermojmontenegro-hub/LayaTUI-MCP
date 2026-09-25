# Herramientas locales para Laya

Este proyecto instala una TUI y un servidor MCP para decisiones de agentes.
Usa `laya==0.3.20` desde PyPI; no contiene ni necesita el repositorio upstream.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python .
.venv/bin/laya-tui
LAYA_PRELOAD=0 .venv/bin/laya-agent-mcp
```

La TUI permite enviar texto o estado JSON, elegir preguntas predefinidas o
personalizadas, inspeccionar rutas, ver respuestas resumidas o JSON y guardar
el último resultado. El selector de dispositivo admite CPU, CUDA, MPS, XPU y
un identificador personalizado como `cuda:1`. Los modelos se descargan la
primera vez que se usan.

`laya-agent-mcp` ofrece `laya_status`, `laya_route`, `laya_predict`,
`laya_preset`, `laya_shortlist` y `laya_choose_action`, además del prompt
`laya_agent_decisions`. La última herramienta elige entre acciones concretas
de browser/computer use y no las ejecuta. Más de 20 acciones se reducen con
embeddings antes de elegir. Para decisiones confiables de navegación, configure
`LAYA_ACTION_MODEL` con un checkpoint local entrenado para el esquema
`next_action`; los checkpoints generales todavía no están validados para ello.

El servidor usa `LAYA_DEVICE=cpu` o `LAYA_DEVICE=cuda` para seleccionar el
dispositivo y `LAYA_PRELOAD=0` para cargar los modelos al primer uso.
