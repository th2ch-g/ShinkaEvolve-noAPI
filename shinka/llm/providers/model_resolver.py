from dataclasses import dataclass
from typing import Optional

from shinka.agent_cli_config import parse_agent_cli_model
from shinka.local_openai_config import parse_local_openai_model
from .pricing import get_provider

_OPENROUTER_PREFIX = "openrouter/"


@dataclass(frozen=True)
class ResolvedModel:
    original_model_name: str
    api_model_name: str
    provider: str
    base_url: Optional[str] = None
    api_key_env_name: Optional[str] = None
    agent_name: Optional[str] = None
    cli_model_name: Optional[str] = None


def resolve_model_backend(model_name: str) -> ResolvedModel:
    """Resolve runtime backend info for known and dynamic model identifiers."""
    provider = get_provider(model_name)
    if provider is not None:
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=model_name,
            provider=provider,
            base_url=None,
        )

    if model_name.startswith("azure-"):
        api_model_name = model_name.split("azure-", 1)[-1]
        if not api_model_name:
            raise ValueError("Azure model name is missing after 'azure-' prefix.")
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="azure_openai",
            base_url=None,
        )

    if model_name.startswith(_OPENROUTER_PREFIX):
        api_model_name = model_name.split(_OPENROUTER_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError("OpenRouter model name is missing after 'openrouter/'.")
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="openrouter",
            base_url=None,
        )

    local_match = parse_local_openai_model(model_name)
    if local_match:
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=local_match.api_model_name,
            provider="local_openai",
            base_url=local_match.base_url,
            api_key_env_name=local_match.api_key_env_name,
        )

    agent_match = parse_agent_cli_model(model_name)
    if agent_match:
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=agent_match.display_model_name,
            provider=agent_match.provider,
            base_url=None,
            agent_name=agent_match.agent_name,
            cli_model_name=agent_match.cli_model_name,
        )

    raise ValueError(
        f"Model '{model_name}' is not supported. "
        "Use a known pricing.csv model, 'openrouter/<model>', "
        "'local/<model>@http(s)://host[:port]/v1', "
        "'claude-code[/<model>]', or 'codex[/<model>]'."
    )
