from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from typing import Optional


AGENT_CLI_PROVIDERS = ("claude_code", "codex_cli")

_AGENT_PREFIXES: dict[str, tuple[str, str]] = {
    "claude-code": ("claude_code", "claude-code"),
    "claude_code": ("claude_code", "claude-code"),
    "codex": ("codex_cli", "codex"),
}

_DEFAULT_COMMANDS: dict[str, str] = {
    "claude_code": "claude",
    "codex_cli": "codex",
}

_COMMAND_ENV_VARS: dict[str, str] = {
    "claude_code": "SHINKA_CLAUDE_CODE_COMMAND",
    "codex_cli": "SHINKA_CODEX_COMMAND",
}

_EXTRA_ARGS_ENV_VARS: dict[str, str] = {
    "claude_code": "SHINKA_CLAUDE_CODE_ARGS",
    "codex_cli": "SHINKA_CODEX_ARGS",
}


@dataclass(frozen=True)
class ResolvedAgentCLIModel:
    original_model_name: str
    provider: str
    agent_name: str
    cli_model_name: Optional[str] = None

    @property
    def display_model_name(self) -> str:
        if self.cli_model_name:
            return f"{self.agent_name}/{self.cli_model_name}"
        return self.agent_name


def parse_agent_cli_model(model_name: str) -> Optional[ResolvedAgentCLIModel]:
    """Parse Claude Code / Codex CLI model identifiers.

    Supported forms:
    - claude-code
    - claude-code/<model>
    - codex
    - codex/<model>
    """
    normalized_model_name = model_name.strip()
    for prefix, (provider, agent_name) in _AGENT_PREFIXES.items():
        if normalized_model_name == prefix:
            return ResolvedAgentCLIModel(
                original_model_name=model_name,
                provider=provider,
                agent_name=agent_name,
            )

        prefix_with_separator = f"{prefix}/"
        if normalized_model_name.startswith(prefix_with_separator):
            cli_model_name = normalized_model_name[len(prefix_with_separator) :].strip()
            if not cli_model_name:
                raise ValueError(
                    f"Agent CLI model name is missing after '{prefix_with_separator}'."
                )
            return ResolvedAgentCLIModel(
                original_model_name=model_name,
                provider=provider,
                agent_name=agent_name,
                cli_model_name=cli_model_name,
            )

    return None


def is_agent_cli_provider(provider: str) -> bool:
    return provider in AGENT_CLI_PROVIDERS


def agent_cli_command_parts(provider: str) -> tuple[str, ...]:
    env_var_name = _COMMAND_ENV_VARS[provider]
    command = os.getenv(env_var_name, _DEFAULT_COMMANDS[provider]).strip()
    if not command:
        raise ValueError(f"{env_var_name} must not be empty.")
    return tuple(shlex.split(command))


def agent_cli_extra_args(provider: str) -> tuple[str, ...]:
    extra_args = os.getenv(_EXTRA_ARGS_ENV_VARS[provider], "").strip()
    if not extra_args:
        return ()
    return tuple(shlex.split(extra_args))


def agent_cli_executable_exists(provider: str) -> bool:
    command_parts = agent_cli_command_parts(provider)
    return shutil.which(command_parts[0]) is not None

