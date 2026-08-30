"""Ollama Local LLM Provider implementation."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from amea.llm.base import LLMProviderBase, LLMResponse, ProviderStatus, ProviderType


class OllamaProvider(LLMProviderBase):
    """Integrates with a local or remote Ollama server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        raw_base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        self.base_url = raw_base.rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3").strip()

    def health_check(self) -> ProviderStatus:
        """Check if local Ollama server is reachable and inspect installed models."""
        try:
            req = urllib.request.Request(
                url=f"{self.base_url}/api/tags",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    model_found = any(self.model in m for m in models) if models else False
                    msg = (
                        f"Ollama server reachable at {self.base_url}. Model '{self.model}' "
                        f"{'is installed' if model_found else 'NOT found in local models: ' + str(models[:3])}."
                    )
                    return ProviderStatus(
                        provider_type=ProviderType.OLLAMA,
                        model_name=self.model,
                        is_available=True,
                        status_message=msg,
                        base_url=self.base_url,
                        has_api_key=False,
                        metadata={"available_models": models},
                    )
        except Exception as e:
            return ProviderStatus(
                provider_type=ProviderType.OLLAMA,
                model_name=self.model,
                is_available=False,
                status_message=f"Ollama server unreachable at {self.base_url} ({type(e).__name__})",
                base_url=self.base_url,
                has_api_key=False,
                metadata={"error": str(e)},
            )

        return ProviderStatus(
            provider_type=ProviderType.OLLAMA,
            model_name=self.model,
            is_available=False,
            status_message=f"Ollama server unreachable at {self.base_url}",
            base_url=self.base_url,
            has_api_key=False,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call Ollama /api/chat endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider=ProviderType.OLLAMA,
                    usage={"total_duration": data.get("total_duration", 0)},
                    raw_response=data,
                )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}. Please ensure Ollama is running or configure OpenRouter."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {str(e)}") from e
