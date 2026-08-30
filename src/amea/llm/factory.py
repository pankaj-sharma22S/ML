"""Factory for managing and instantiating configured LLM Providers."""

import os
from typing import Any, Dict, List, Optional

from amea.llm.base import LLMProviderBase, ProviderStatus, ProviderType
from amea.llm.ollama import OllamaProvider
from amea.llm.openrouter import OpenRouterProvider


class LLMProviderFactory:
    """Manages active LLM provider selection, health status, and fallback mechanisms."""

    @classmethod
    def get_provider(cls, provider_type: Optional[str] = None) -> LLMProviderBase:
        """Instantiate requested or default provider."""
        ptype = (provider_type or os.environ.get("LLM_PROVIDER", "")).strip().lower()

        if ptype == ProviderType.OPENROUTER.value or (not ptype and os.environ.get("OPENROUTER_API_KEY")):
            return OpenRouterProvider()
        
        if ptype == ProviderType.OLLAMA.value:
            return OllamaProvider()

        # Default heuristic: if OPENROUTER_API_KEY is present, use OpenRouter, else Ollama
        if os.environ.get("OPENROUTER_API_KEY"):
            return OpenRouterProvider()
        
        return OllamaProvider()

    @classmethod
    def get_all_status(cls) -> Dict[str, ProviderStatus]:
        """Check and return status for both OpenRouter and Ollama."""
        openrouter = OpenRouterProvider()
        ollama = OllamaProvider()
        return {
            ProviderType.OPENROUTER.value: openrouter.health_check(),
            ProviderType.OLLAMA.value: ollama.health_check(),
        }

    @classmethod
    def get_active_status(cls) -> Dict[str, Any]:
        """Return the active provider's status and system summary."""
        active = cls.get_provider()
        health = active.health_check()
        all_status = cls.get_all_status()
        
        return {
            "active_provider": health.provider_type.value,
            "active_model": health.model_name,
            "is_available": health.is_available,
            "status_message": health.status_message,
            "providers": {k: v.model_dump() for k, v in all_status.items()},
        }
