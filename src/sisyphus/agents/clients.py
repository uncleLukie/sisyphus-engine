"""Chat client factory: Microsoft Foundry first, with Azure OpenAI / OpenAI fallbacks.

Resolution order (first match wins):

1. ``FOUNDRY_PROJECT_ENDPOINT`` + ``FOUNDRY_MODEL`` -> :class:`FoundryChatClient`
   (Entra ID via ``DefaultAzureCredential``; use ``az login`` locally and a
   managed identity on Azure Container Apps).
2. ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_MODEL`` -> Azure OpenAI through
   :class:`OpenAIChatClient` (API key if ``AZURE_OPENAI_API_KEY`` is set,
   otherwise Entra ID).
3. ``OPENAI_API_KEY`` -> OpenAI.
4. Nothing configured -> ``None``; the workflow runs in deterministic template
   mode so the whole pipeline works offline and in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ClientInfo:
    provider: str  # foundry | azure_openai | openai | template
    model: str | None

    @property
    def available(self) -> bool:
        return self.provider != "template"


def describe_client() -> ClientInfo:
    if os.getenv("SISYPHUS_FORCE_TEMPLATE", "").lower() in ("1", "true", "yes"):
        return ClientInfo("template", None)
    if os.getenv("FOUNDRY_PROJECT_ENDPOINT") and os.getenv("FOUNDRY_MODEL"):
        return ClientInfo("foundry", os.environ["FOUNDRY_MODEL"])
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_MODEL"):
        return ClientInfo("azure_openai", os.environ["AZURE_OPENAI_MODEL"])
    if os.getenv("OPENAI_API_KEY"):
        return ClientInfo("openai", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    return ClientInfo("template", None)


def make_chat_client() -> Any | None:
    """Build the configured chat client, or ``None`` for template mode."""
    info = describe_client()
    if info.provider == "foundry":
        from agent_framework.foundry import FoundryChatClient
        from azure.identity import DefaultAzureCredential

        return FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=DefaultAzureCredential(),
        )
    if info.provider == "azure_openai":
        from agent_framework.openai import OpenAIChatClient

        kwargs: dict[str, Any] = {
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "model": os.environ["AZURE_OPENAI_MODEL"],
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        }
        if os.getenv("AZURE_OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        else:
            from azure.identity import DefaultAzureCredential

            kwargs["credential"] = DefaultAzureCredential()
        return OpenAIChatClient(**kwargs)
    if info.provider == "openai":
        from agent_framework.openai import OpenAIChatClient

        return OpenAIChatClient(api_key=os.environ["OPENAI_API_KEY"], model=info.model)
    return None
