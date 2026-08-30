"""Base interfaces and data models for LLM providers in AMEA."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    MOCK = "mock"


class ProviderStatus(BaseModel):
    provider_type: ProviderType
    model_name: str
    is_available: bool
    status_message: str
    base_url: str
    has_api_key: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: ProviderType
    usage: Dict[str, int] = Field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None


class LLMProviderBase(ABC):
    """Abstract base class for all LLM providers in AMEA."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response synchronously."""
        pass

    @abstractmethod
    def health_check(self) -> ProviderStatus:
        """Check provider connectivity, authentication, and readiness."""
        pass
