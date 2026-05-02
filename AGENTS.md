# Repository Guidelines

## Project Structure & Module Organization

`shinka/` contains the Python package. Core evolution logic lives in `shinka/core/`, provider integrations in `shinka/llm/` and `shinka/embed/`, persistence code in `shinka/database/`, CLI entry points in `shinka/cli/`, packaged YAML defaults in `shinka/configs/`, and WebUI assets in `shinka/webui/`. Tests are in `tests/` and generally mirror the behavior they cover with `test_*.py` files. Documentation lives in `docs/` and is built by MkDocs. Runnable demos and benchmark-style examples are under `examples/`. Agent skills are stored in `skills/`.

## Build, Test, and Development Commands

Use `uv` for local Python workflows.

```bash
uv sync --dev
```

Installs the project with development dependencies.

```bash
uv run pytest -q -m "not requires_secrets"
```

Runs the normal test suite while skipping secret-backed provider tests.

```bash
uv run ruff check tests --exclude tests/file.py
uv run mypy --follow-imports=skip --ignore-missing-imports tests/test_*.py tests/conftest.py
```

Runs the repository's lint and type checks from `CONTRIBUTING.md`.

```bash
uv sync --group docs
uv run --group docs mkdocs serve --dev-addr 127.0.0.1:8000
```

Serves the documentation locally.

## Coding Style & Naming Conventions

Target Python 3.10+. Use Black-compatible formatting, Ruff linting, and 4-space indentation. Prefer `snake_case` for modules, functions, variables, and test files; use `PascalCase` for classes. Keep public APIs typed where practical and keep provider-specific behavior isolated in the relevant `llm` or `embed` backend module.

## Testing Guidelines

Add or update tests for bug fixes, behavior changes, and regressions. Name tests `test_<behavior>.py` or `test_<specific_case>` inside an existing file. Use pytest markers defined in `pyproject.toml`: `requires_secrets` for tests that need credentials and `integration` for live external coverage. For coverage checks, run:

```bash
uv run --with pytest-cov pytest -q -m "not requires_secrets" --cov=shinka --cov-report=term-missing
```

## Commit & Pull Request Guidelines

History uses concise Conventional Commit-style subjects such as `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, and `refactor: ...`. Keep commits focused and describe user-visible behavior. Pull requests should include a short summary, motivation, linked issues, tests run, and risks. Include docs or screenshots when changing CLI output, documentation, examples, or WebUI behavior.

## Security & Configuration Tips

Do not commit API keys, provider credentials, generated databases, or local `.env` files. Keep secret-backed validation behind `requires_secrets`, and document any required environment variables near the code or example that uses them.
