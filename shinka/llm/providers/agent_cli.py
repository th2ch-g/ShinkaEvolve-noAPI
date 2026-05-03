from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shinka.agent_cli_config import (
    agent_cli_command_parts,
    agent_cli_extra_args,
    agent_cli_executable_exists,
)
from shinka.llm.constants import TIMEOUT

from .result import QueryResult

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 4000
_TOKEN_LIMIT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btoken(?:s)?\s+(?:limit|quota|budget)\b",
        r"\b(?:limit|quota|budget)\s+(?:for\s+)?token(?:s)?\b",
        r"\btoken(?:s)?.*(?:reached|exceed(?:s|ed|ing)?)\b",
        r"\b(?:reached|exceed(?:s|ed|ing)?).*token(?:s)?\b",
        r"\btoo many tokens\b",
        r"\bmaximum context length\b",
        r"\bcontext (?:length|window|limit)\b",
        r"\bcontext .* exceeded\b",
        r"\bexceed(?:s|ed|ing)? .* context\b",
        r"\busage limit (?:reached|exceeded)\b",
        r"\bquota (?:reached|exceeded)\b",
        r"\brate limit (?:reached|exceeded)\b",
        r"\bbilling .* limit\b",
    )
)


class AgentCLITokenLimitError(RuntimeError):
    """Raised when a local agent CLI reports an account/context token limit."""

    def __init__(
        self,
        agent_name: str,
        returncode: int,
        details: str,
    ) -> None:
        self.agent_name = agent_name
        self.returncode = returncode
        self.details = details
        super().__init__(
            f"{agent_name} CLI token or usage limit reached "
            f"(exit code {returncode}): {details}"
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.agent_name, self.returncode, self.details),
        )


@dataclass(frozen=True)
class AgentCLIClient:
    provider: str
    agent_name: str
    cli_model_name: Optional[str]
    command_parts: tuple[str, ...]
    extra_args: tuple[str, ...]
    timeout: float
    cwd: str


def build_agent_cli_client(resolved_model) -> AgentCLIClient:
    """Build a lightweight client config for a local coding-agent CLI."""
    provider = resolved_model.provider
    agent_name = resolved_model.agent_name
    if agent_name is None:
        raise ValueError(f"Missing agent name for provider '{provider}'.")
    if not agent_cli_executable_exists(provider):
        command = " ".join(agent_cli_command_parts(provider))
        raise ValueError(
            f"Agent CLI backend '{agent_name}' requires the "
            f"'{command}' executable on PATH. Install/login to the CLI or set the "
            "matching SHINKA_*_COMMAND environment variable."
        )

    timeout = float(os.getenv("SHINKA_AGENT_CLI_TIMEOUT", str(TIMEOUT)))
    cwd = os.getenv("SHINKA_AGENT_CLI_CWD", os.getcwd())
    return AgentCLIClient(
        provider=provider,
        agent_name=agent_name,
        cli_model_name=resolved_model.cli_model_name,
        command_parts=agent_cli_command_parts(provider),
        extra_args=agent_cli_extra_args(provider),
        timeout=timeout,
        cwd=cwd,
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(_content_to_text(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
    return str(content)


def _render_history(msg_history: List[Dict]) -> str:
    rendered_messages = []
    for message in msg_history:
        role = str(message.get("role", "message")).capitalize()
        content = _content_to_text(message.get("content")).strip()
        if content:
            rendered_messages.append(f"{role}:\n{content}")
    return "\n\n".join(rendered_messages)


def _render_prompt(
    *,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    include_system_msg: bool,
) -> str:
    sections = []
    if include_system_msg and system_msg:
        sections.append(f"System instructions:\n{system_msg}")

    history = _render_history(msg_history)
    if history:
        sections.append(f"Conversation history:\n{history}")

    sections.append(f"User request:\n{msg}")
    sections.append(
        "Return only the response requested by Shinka. Do not modify files or "
        "run project commands."
    )
    return "\n\n".join(sections)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return text[:_MAX_ERROR_CHARS] + "\n...[truncated]"


def _looks_like_token_limit(details: str) -> bool:
    return any(pattern.search(details) for pattern in _TOKEN_LIMIT_PATTERNS)


def _run_process(
    args: list[str],
    *,
    prompt: str,
    timeout: float,
    cwd: str,
) -> subprocess.CompletedProcess[str]:
    logger.debug("Running agent CLI command: %s", " ".join(args))
    return subprocess.run(
        args,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
        check=False,
    )


def _raise_for_failure(
    completed: subprocess.CompletedProcess[str],
    *,
    agent_name: str,
) -> None:
    if completed.returncode == 0:
        return
    stdout = _truncate(completed.stdout.strip())
    stderr = _truncate(completed.stderr.strip())
    details = stderr or stdout or "no output"
    combined_output = "\n".join(part for part in [stderr, stdout] if part)
    if _looks_like_token_limit(combined_output or details):
        raise AgentCLITokenLimitError(
            agent_name=agent_name,
            returncode=completed.returncode,
            details=details,
        )
    raise RuntimeError(
        f"{agent_name} CLI query failed with exit code "
        f"{completed.returncode}: {details}"
    )


def _run_claude_code(client: AgentCLIClient, prompt: str, system_msg: str) -> str:
    args = [
        *client.command_parts,
        "-p",
        "--output-format",
        "text",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "",
    ]
    if client.cli_model_name:
        args.extend(["--model", client.cli_model_name])
    if system_msg:
        args.extend(["--system-prompt", system_msg])
    args.extend(client.extra_args)

    completed = _run_process(
        args,
        prompt=prompt,
        timeout=client.timeout,
        cwd=client.cwd,
    )
    _raise_for_failure(completed, agent_name=client.agent_name)
    return completed.stdout.strip()


def _run_codex(client: AgentCLIClient, prompt: str) -> str:
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as handle:
            output_path = handle.name

        args = [
            *client.command_parts,
            "exec",
            "--cd",
            client.cwd,
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "--output-last-message",
            output_path,
        ]
        if client.cli_model_name:
            args.extend(["--model", client.cli_model_name])
        args.extend(client.extra_args)
        args.append("-")

        completed = _run_process(
            args,
            prompt=prompt,
            timeout=client.timeout,
            cwd=client.cwd,
        )
        _raise_for_failure(completed, agent_name=client.agent_name)

        content = Path(output_path).read_text(encoding="utf-8").strip()
        return content or completed.stdout.strip()
    finally:
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove Codex output file %s", output_path)


def query_agent_cli(
    client: AgentCLIClient,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query a local Claude Code or Codex CLI as an LLM backend."""
    if output_model is not None:
        raise NotImplementedError(
            "Structured output is not supported for agent CLI backends."
        )

    include_system_msg = client.provider == "codex_cli"
    prompt = _render_prompt(
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        include_system_msg=include_system_msg,
    )

    if client.provider == "claude_code":
        content = _run_claude_code(client, prompt, system_msg)
    elif client.provider == "codex_cli":
        content = _run_codex(client, prompt)
    else:
        raise ValueError(f"Unsupported agent CLI provider: {client.provider}")

    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=_estimate_tokens(prompt),
        output_tokens=_estimate_tokens(content),
        thinking_tokens=0,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought="",
        model_posteriors=model_posteriors,
    )


async def query_agent_cli_async(
    client: AgentCLIClient,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    return await asyncio.to_thread(
        query_agent_cli,
        client,
        model,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
