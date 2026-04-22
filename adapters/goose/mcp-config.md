# Registering `rlm-mcp-server` with Goose

Goose's MCP extensions live in the user's global config file:

- **macOS / Linux**: `~/.config/goose/config.yaml`
- **Windows**: `%APPDATA%\Block\goose\config\config.yaml`

Source: <https://goose-docs.ai/docs/guides/config-files/>.

Goose uses **map-keyed** extensions (not a list): the key under `extensions:`
is the extension handle you reference elsewhere. Do NOT use a list-of-objects
here — that is the *recipe*-level schema, which is different.

## 1. YAML block to add

Open `~/.config/goose/config.yaml` and add the following under the top-level
`extensions:` key (create it if it doesn't exist):

```yaml
extensions:
  rlm-mcp-server:
    type: stdio
    name: rlm-mcp-server
    enabled: true
    bundled: false
    cmd: uv
    args:
      - run
      - --with
      - anthropic
      - --with
      - mcp
      - python
      - -m
      - harness_rlm.mcp_server
    env_keys:
      - ANTHROPIC_API_KEY
    timeout: 300
    description: >-
      RLM sub-LLM dispatcher. Exposes a single tool `llm_query(prompt, model,
      max_tokens?, system?)` that calls the Anthropic API directly. Used by the
      `rlm` recipe to fan out cheap per-chunk queries without paying Goose's
      per-subagent process-boot overhead.
```

### Notes on the fields

| Field       | Why it looks like this                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `type: stdio` | Goose's MCP-over-stdio transport. The server in `src/harness_rlm/mcp_server.py` uses `mcp.server.stdio.stdio_server`. |
| `cmd: uv`   | Assumes `uv` is on `$PATH`. Swap to an absolute path (e.g. `/opt/homebrew/bin/uv`) if Goose launches without your shell env. |
| `args`      | Uses `python -m harness_rlm.mcp_server`, so you must have run `uv pip install -e .` (or `uv sync`) inside `/Users/rshah/harness-rlm` at least once so the package is importable. |
| `env_keys`  | Goose passes `ANTHROPIC_API_KEY` through from the process environment (no value stored on disk). Set it before launching Goose. |
| `bundled: false` | This is a user-registered extension, not one shipped with Goose itself. |

### Alternative: pin to a checked-out repo

If you want Goose to launch the server *without* needing the package installed
site-wide, replace the args with a direct script path:

```yaml
    cmd: uv
    args:
      - run
      - --with
      - anthropic
      - --with
      - mcp
      - python
      - /Users/rshah/harness-rlm/src/harness_rlm/mcp_server.py
```

## 2. Export the Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Add to ~/.zshrc or ~/.bashrc to persist across terminals.
```

Goose inherits the env of the shell that launches it, so whichever shell you
run `goose run` from must have the key exported.

## 3. Smoke-test

The MCP server ships a self-test:

```bash
ANTHROPIC_API_KEY="sk-ant-..." \
  uv run --with anthropic --with mcp \
  python -m harness_rlm.mcp_server --selftest
```

Expected: one `llm_query` round-trip, a cost line, `[selftest] PASS`.

Once the server itself passes, confirm Goose can see it by running the root
recipe in render mode (no LLM calls, no tool dispatch — just templating):

```bash
goose run \
  --recipe /Users/rshah/harness-rlm/adapters/goose/recipes/rlm.yaml \
  --params user_query="hello" \
  --render-recipe
```

`--render-recipe` prints the fully-rendered YAML and exits without executing,
per the `goose run` reference (<https://goose-docs.ai/docs/guides/goose-cli-commands>).
If the rendered YAML includes the `rlm-mcp-server` extension block, Goose has
loaded the recipe correctly.

A full end-to-end headless run looks like:

```bash
goose run \
  --recipe ~/.config/goose/recipes/rlm.yaml \
  --params user_query="Summarise the attached doc" \
  --params context_path="/path/to/long_doc.md" \
  --no-session \
  --output-format json
```

## 4. Troubleshooting

- **`uv: command not found` in Goose logs** → Goose launched without your
  shell's PATH. Use the absolute path to `uv` in `cmd:`.
- **`ANTHROPIC_API_KEY is not set`** → the key didn't propagate. Either export
  it before launching Goose, or put it in `~/.config/goose/secrets.yaml` under
  the `rlm-mcp-server` key (Goose's file-based secret store).
- **Tool not visible to the recipe** → extensions registered in the **recipe's**
  `extensions:` list are per-recipe and override the global config. The recipe
  here declares the same extension so it works even without the global block,
  but both must point to the same `cmd/args` or you'll get two disagreeing
  servers.
