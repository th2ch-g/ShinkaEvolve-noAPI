from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
import shinka.llm.providers.agent_cli as agent_cli_module
from shinka.llm import AsyncLLMClient
from shinka.llm.providers.agent_cli import (
    AgentCLITokenLimitError,
    AgentCLIClient,
    build_agent_cli_client,
    query_agent_cli,
)


def test_build_agent_cli_client_uses_env_command_and_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("SHINKA_CODEX_COMMAND", sys.executable)
    monkeypatch.setenv("SHINKA_AGENT_CLI_CWD", str(tmp_path))

    client = build_agent_cli_client(
        SimpleNamespace(
            provider="codex_cli",
            agent_name="codex",
            cli_model_name="gpt-5.4-mini",
        )
    )

    assert client.command_parts == (sys.executable,)
    assert client.cwd == str(tmp_path)
    assert client.cli_model_name == "gpt-5.4-mini"


def test_query_agent_cli_returns_zero_cost_result(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_codex(client, prompt):
        captured["prompt"] = prompt
        return "<diff>patch</diff>"

    monkeypatch.setattr(agent_cli_module, "_run_codex", _fake_run_codex)
    client = AgentCLIClient(
        provider="codex_cli",
        agent_name="codex",
        cli_model_name=None,
        command_parts=("codex",),
        extra_args=(),
        timeout=1200,
        cwd=str(tmp_path),
    )

    result = query_agent_cli(
        client,
        "codex",
        msg="make it faster",
        system_msg="Preserve correctness.",
        msg_history=[{"role": "assistant", "content": "previous answer"}],
        output_model=None,
        temperature=0.5,
    )

    assert result.content == "<diff>patch</diff>"
    assert result.cost == 0.0
    assert result.model_name == "codex"
    assert result.kwargs["temperature"] == 0.5
    assert "System instructions:" in captured["prompt"]
    assert "previous answer" in captured["prompt"]
    assert result.new_msg_history[-1] == {
        "role": "assistant",
        "content": "<diff>patch</diff>",
    }


def test_run_codex_reads_last_message_file(monkeypatch, tmp_path):
    captured = {}
    client = AgentCLIClient(
        provider="codex_cli",
        agent_name="codex",
        cli_model_name="gpt-5.4-mini",
        command_parts=("codex",),
        extra_args=("--debug",),
        timeout=1200,
        cwd=str(tmp_path),
    )

    def _fake_run_process(args, *, prompt, timeout, cwd):
        captured["args"] = args
        output_path = Path(args[args.index("--output-last-message") + 1])
        output_path.write_text("final response\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="logs", stderr="")

    monkeypatch.setattr(agent_cli_module, "_run_process", _fake_run_process)

    content = agent_cli_module._run_codex(client, "prompt")

    assert content == "final response"
    assert captured["args"][:2] == ["codex", "exec"]
    assert "--model" in captured["args"]
    assert "gpt-5.4-mini" in captured["args"]
    assert "--debug" in captured["args"]
    assert "--ignore-user-config" in captured["args"]
    assert "--ignore-rules" in captured["args"]
    assert captured["args"][-1] == "-"


def test_run_claude_code_uses_print_mode_and_system_prompt(monkeypatch, tmp_path):
    captured = {}
    client = AgentCLIClient(
        provider="claude_code",
        agent_name="claude-code",
        cli_model_name="sonnet",
        command_parts=("claude",),
        extra_args=("--debug",),
        timeout=1200,
        cwd=str(tmp_path),
    )

    def _fake_run_process(args, *, prompt, timeout, cwd):
        captured["args"] = args
        captured["prompt"] = prompt
        return subprocess.CompletedProcess(args, 0, stdout="final\n", stderr="")

    monkeypatch.setattr(agent_cli_module, "_run_process", _fake_run_process)

    content = agent_cli_module._run_claude_code(client, "prompt", "system")

    assert content == "final"
    assert captured["prompt"] == "prompt"
    assert captured["args"][:2] == ["claude", "-p"]
    assert "--system-prompt" in captured["args"]
    assert "system" in captured["args"]
    assert "--model" in captured["args"]
    assert "sonnet" in captured["args"]
    assert "--debug" in captured["args"]


def test_raise_for_failure_detects_agent_cli_token_limit():
    completed = subprocess.CompletedProcess(
        ["codex"],
        1,
        stdout="",
        stderr="usage limit reached for this account",
    )

    with pytest.raises(AgentCLITokenLimitError) as exc_info:
        agent_cli_module._raise_for_failure(completed, agent_name="codex")

    assert exc_info.value.agent_name == "codex"
    assert exc_info.value.returncode == 1
    assert "usage limit reached" in exc_info.value.details


def test_async_llm_client_does_not_retry_agent_cli_token_limit(monkeypatch):
    calls = 0

    async def _fake_query_async(**kwargs):
        nonlocal calls
        calls += 1
        raise AgentCLITokenLimitError(
            agent_name="codex",
            returncode=1,
            details="usage limit reached",
        )

    monkeypatch.setattr(llm_module, "query_async", _fake_query_async)
    client = AsyncLLMClient(model_names=["codex"], verbose=False)

    with pytest.raises(AgentCLITokenLimitError):
        asyncio.run(client.query(msg="mutate", system_msg="system"))

    assert calls == 1
