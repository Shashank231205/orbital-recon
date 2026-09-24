"""Google Gemini provider."""

import time

import httpx

from orbital_recon.core.exceptions import LLMProviderError
from orbital_recon.llm.base import ChatMessage, CompletionResult, LLMProvider

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """Calls the Gemini generateContent endpoint over HTTP.

    The REST API is used directly rather than the vendor SDK to keep the
    dependency surface small and the failure modes explicit.
    """

    name = "gemini"

    def __init__(self, api_key: str | None, model: str, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(base_url=_BASE_URL, timeout=timeout)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _to_payload(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> dict[str, object]:
        """Convert messages to Gemini's schema.

        Gemini has no system role: system guidance is passed separately as
        ``systemInstruction``, and assistant turns are labelled ``model``.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]

        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        return payload

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        if not self.is_configured:
            raise LLMProviderError("Gemini API key is not configured")

        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"/models/{self._model}:generateContent",
                params={"key": self._api_key},
                json=self._to_payload(messages, temperature, max_tokens),
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Gemini returned {exc.response.status_code}",
                details={"status": exc.response.status_code, "body": exc.response.text[:500]},
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Gemini request failed: {exc}") from exc

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            # A response blocked by safety filters has no parts; treat it as a
            # provider failure so the router can try the next backend.
            raise LLMProviderError(
                "Gemini returned no usable content", details={"body": str(body)[:500]}
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
            response = await self._client.get(
                f"/models/{self._model}", params={"key": self._api_key}
            )
            return response.status_code == httpx.codes.OK
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
