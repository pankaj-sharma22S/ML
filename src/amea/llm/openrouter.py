"""OpenRouter LLM Provider implementation with strict key isolation."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from amea.llm.base import LLMProviderBase, LLMResponse, ProviderStatus, ProviderType


class OpenRouterProvider(LLMProviderBase):
    """Integrates with OpenRouter API using strictly environment-configured credentials."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct").strip()
        raw_base = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        self.base_url = raw_base.rstrip("/")

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key and len(self._api_key) > 5)

    def masked_key(self) -> str:
        if not self._api_key:
            return "NOT_CONFIGURED"
        if len(self._api_key) <= 8:
            return "****"
        return f"{self._api_key[:4]}...{self._api_key[-4:]}"

    def health_check(self) -> ProviderStatus:
        """Verify OpenRouter key presence and connectivity."""
        if not self.has_api_key:
            return ProviderStatus(
                provider_type=ProviderType.OPENROUTER,
                model_name=self.model,
                is_available=False,
                status_message="OPENROUTER_API_KEY environment variable is not configured.",
                base_url=self.base_url,
                has_api_key=False,
            )

        # Fast connectivity check to models endpoint
        try:
            req = urllib.request.Request(
                url=f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://amea.ai",
                    "X-Title": "AMEA Autonomous ML Workspace",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return ProviderStatus(
                        provider_type=ProviderType.OPENROUTER,
                        model_name=self.model,
                        is_available=True,
                        status_message=f"OpenRouter active (Key: {self.masked_key()})",
                        base_url=self.base_url,
                        has_api_key=True,
                    )
        except Exception as e:
            return ProviderStatus(
                provider_type=ProviderType.OPENROUTER,
                model_name=self.model,
                is_available=False,
                status_message=f"OpenRouter connection check failed: {type(e).__name__}",
                base_url=self.base_url,
                has_api_key=True,
                metadata={"error_type": type(e).__name__},
            )

        return ProviderStatus(
            provider_type=ProviderType.OPENROUTER,
            model_name=self.model,
            is_available=True,
            status_message=f"OpenRouter configured with key: {self.masked_key()}",
            base_url=self.base_url,
            has_api_key=True,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call OpenRouter chat completions endpoint."""
        if not self.has_api_key:
            raise ValueError(
                "Cannot generate text with OpenRouter: OPENROUTER_API_KEY is not set. "
                "Please configure OPENROUTER_API_KEY in your .env file or environment."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": "https://amea.ai",
                "X-Title": "AMEA Autonomous ML Workspace",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider=ProviderType.OPENROUTER,
                    usage=usage,
                    raw_response=data,
                )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter API error (HTTP {e.code}): {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"OpenRouter request failed: {str(e)}") from e
