from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import shinka.llm.providers.agent_cli as agent_cli_module
from shinka.llm.providers.agent_cli import (
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
        extra_args=("--ignore-rules",),
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

