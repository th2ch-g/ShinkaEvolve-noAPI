# Local, OpenRouter, and Agent CLI Models

Shinka supports dynamic LLM backend routing in `LLMClient` and `AsyncLLMClient`.
It also supports dynamic embedding backend routing in `EmbeddingClient` and
`AsyncEmbeddingClient`.
You can use:

- models listed in the provider pricing CSVs (existing behavior)
- dynamic OpenRouter model IDs
- local OpenAI-compatible servers via inline endpoint URIs
- local Claude Code / Codex CLI backends

---

## Supported Model Name Formats

### 1) Known models (from `pricing.csv`)

```yaml
evo_config:
  llm_models:
    - gpt-5-mini
    - claude-sonnet-4-6
```

### 2) Dynamic OpenRouter models

Prefix with `openrouter/`:

```yaml
evo_config:
  llm_models:
    - openrouter/qwen/qwen3-coder
    - openrouter/deepseek/deepseek-r1
```

Set env var:

```bash
OPENROUTER_API_KEY=...
```

### 3) Local OpenAI-compatible models

Use `local/<model>@<http(s)://endpoint>`:

```yaml
evo_config:
  llm_models:
    - local/qwen2.5-coder@http://localhost:11434/v1
```

Set optional env var:

```bash
LOCAL_OPENAI_API_KEY=local
```

If not set, Shinka uses `"local"` as a default token.

For a per-model custom key env var, append `api_key_env` to the endpoint URL:

```yaml
evo_config:
  llm_models:
    - local/dummy-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY
```

```bash
CUSTOM_API_KEY=...
```

Shinka strips `api_key_env` from the runtime base URL before creating the client.

### 4) Agent CLI backends

Use a logged-in Claude Code or Codex CLI as the mutation LLM without provider API
keys:

```yaml
evo_config:
  llm_models:
    - codex
  embedding_model: null
```

```yaml
evo_config:
  llm_models:
    - claude-code/sonnet
  embedding_model: null
```

Supported model identifiers:

- `codex`
- `codex/<model>`
- `claude-code`
- `claude-code/<model>`

The optional `<model>` is passed through to the CLI's model flag. These
backends report token estimates and `0.0` cost because usage/cost metadata is
not available from the local CLI invocation.

For `shinka_run`, the convenience preset also disables the default OpenAI
embedding model and defaults proposal concurrency to one CLI invocation at a
time:

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_codex \
  --num_generations 20 \
  --agent-backend codex
```

Environment knobs:

- `SHINKA_CLAUDE_CODE_COMMAND` (default: `claude`)
- `SHINKA_CODEX_COMMAND` (default: `codex`)
- `SHINKA_CLAUDE_CODE_ARGS` / `SHINKA_CODEX_ARGS` for extra CLI flags
- `SHINKA_AGENT_CLI_TIMEOUT` in seconds (default: Shinka LLM timeout)
- `SHINKA_AGENT_CLI_CWD` for the CLI working directory

---

## Local Embeddings

The same inline local format also works for `embedding_model`.

```yaml
evo_config:
  embedding_model: local/text-embeddings-inference@http://localhost:8080/v1
```

You can also use the same `api_key_env` query parameter for embeddings:

```yaml
evo_config:
  embedding_model: local/dummy-embed@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY
```

Common local embedding backends:

- Hugging Face TEI:
  `local/text-embeddings-inference@http://localhost:8080/v1`
- vLLM or another OpenAI-compatible embedding server:
  `local/BAAI/bge-small-en-v1.5@http://localhost:8000/v1`
- Ollama OpenAI-compatible endpoint:
  `local/embeddinggemma@http://localhost:11434/v1`

---

## Notes

- Dynamic OpenRouter/local model IDs are allowed even if not listed in `pricing.csv`.
- If a model has no pricing entry and the provider does not return cost metadata, Shinka records cost as `0.0`.
- Local OpenAI-compatible backend path currently uses chat-completions style calls.
- Local embedding backends use the OpenAI-compatible `/v1/embeddings` path.
- `api_key_env` must reference a single environment variable name, for example `CUSTOM_API_KEY`.
- Structured output is not supported yet for `local/...@...` or agent CLI models.

---

## Applies to Which Clients

These formats work across all LLM consumers that use `LLMClient` / `AsyncLLMClient`, including:

- mutation LLMs (`llm_models`)
- meta LLMs (`meta_llm_models`)
- novelty judge LLMs (`novelty_llm_models`)
- prompt evolution LLMs (`prompt_llm_models`)

For embeddings, the same format applies to:

- code similarity embeddings (`embedding_model`)
