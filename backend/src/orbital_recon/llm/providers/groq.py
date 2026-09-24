"""Groq provider."""

import time

import httpx

from orbital_recon.core.exceptions import LLMProviderError
from orbital_recon.llm.base import ChatMessage, CompletionResult, LLMProvider

_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    """Calls Groq's OpenAI-compatible chat completions endpoint."""

    name = "groq"

    def __init__(self, api_key: str | None, model: str, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        if not self.is_configured:
            raise LLMProviderError("Groq API key is not configured")

        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Groq returned {exc.response.status_code}",
                details={"status": exc.response.status_code, "body": exc.response.text[:500]},
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "Groq returned no usable content", details={"body": str(body)[:500]}
            ) from exc

        return CompletionResult(
            text=text.strip(),
            provider=self.name,
            model=self._model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health_check(self) -> bool:
        if not self.is_configured:
            return False
        try:
            response = await self._client.get("/models")
            return response.status_code == httpx.codes.OK
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
