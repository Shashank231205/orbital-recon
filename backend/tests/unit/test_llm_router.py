"""Tests for language model failover routing."""

import pytest

from orbital_recon.core.config import Settings
from orbital_recon.core.exceptions import AllProvidersUnavailableError, LLMProviderError
from orbital_recon.llm.base import ChatMessage, CompletionResult, LLMProvider
from orbital_recon.llm.router import LLMRouter


class FakeProvider(LLMProvider):
    """A provider whose behaviour is dictated by the test."""

    def __init__(
        self,
        name: str,
        *,
        configured: bool = True,
        fails: bool = False,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self._configured = configured
        self.fails = fails
        self._healthy = healthy
        self.call_count = 0

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        self.call_count += 1
        if self.fails:
            raise LLMProviderError(f"{self.name} is unavailable")
        return CompletionResult(
            text=f"response from {self.name}",
            provider=self.name,
            model="fake",
            latency_ms=1,
        )

    async def health_check(self) -> bool:
        return self._healthy and self._configured


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        groq_api_key="test-key",
        llm_health_ttl_seconds=60.0,
    )


def build_router(settings: Settings, *providers: FakeProvider) -> LLMRouter:
    """Construct a router driven by fake providers."""
    return LLMRouter(settings, providers=providers)


MESSAGES = [ChatMessage(role="user", content="status report")]


class TestRouting:
    async def test_uses_first_healthy_provider(self, settings: Settings) -> None:
        primary = FakeProvider("gemini")
        secondary = FakeProvider("groq")
        router = build_router(settings, primary, secondary)

        result = await router.complete(MESSAGES)

        assert result.provider == "gemini"
        assert secondary.call_count == 0

    async def test_falls_back_when_primary_fails(self, settings: Settings) -> None:
        primary = FakeProvider("gemini", fails=True)
        secondary = FakeProvider("groq")
        router = build_router(settings, primary, secondary)

        result = await router.complete(MESSAGES)

        assert result.provider == "groq"
        assert primary.call_count == 1

    async def test_skips_unconfigured_provider(self, settings: Settings) -> None:
        primary = FakeProvider("gemini", configured=False)
        secondary = FakeProvider("groq")
        router = build_router(settings, primary, secondary)

        result = await router.complete(MESSAGES)

        assert result.provider == "groq"
        assert primary.call_count == 0

    async def test_raises_when_all_providers_fail(self, settings: Settings) -> None:
        router = build_router(
            settings, FakeProvider("gemini", fails=True), FakeProvider("groq", fails=True)
        )

        with pytest.raises(AllProvidersUnavailableError) as exc_info:
            await router.complete(MESSAGES)

        assert exc_info.value.details["attempted"] == ["gemini", "groq"]

    async def test_raises_when_nothing_configured(self, settings: Settings) -> None:
        router = build_router(
            settings,
            FakeProvider("gemini", configured=False),
            FakeProvider("groq", configured=False),
        )

        with pytest.raises(AllProvidersUnavailableError):
            await router.complete(MESSAGES)


class TestHealthCooldown:
    async def test_failed_provider_is_skipped_while_cooling_down(
        self, settings: Settings
    ) -> None:
        """A provider that just failed is not retried on the next request."""
        primary = FakeProvider("gemini", fails=True)
        secondary = FakeProvider("groq")
        router = build_router(settings, primary, secondary)

        await router.complete(MESSAGES)
        await router.complete(MESSAGES)

        assert primary.call_count == 1
        assert secondary.call_count == 2

    async def test_provider_retried_after_cooldown_expires(self, settings: Settings) -> None:
        settings = Settings(
            gemini_api_key="k", groq_api_key="k", llm_health_ttl_seconds=0.0
        )
        primary = FakeProvider("gemini", fails=True)
        router = build_router(settings, primary, FakeProvider("groq"))

        await router.complete(MESSAGES)
        await router.complete(MESSAGES)

        assert primary.call_count == 2

    async def test_recovered_provider_regains_priority(self, settings: Settings) -> None:
        settings = Settings(
            gemini_api_key="k", groq_api_key="k", llm_health_ttl_seconds=0.0
        )
        primary = FakeProvider("gemini", fails=True)
        router = build_router(settings, primary, FakeProvider("groq"))

        assert (await router.complete(MESSAGES)).provider == "groq"
        primary.fails = False
        assert (await router.complete(MESSAGES)).provider == "gemini"


class TestStatus:
    async def test_refresh_reports_each_provider(self, settings: Settings) -> None:
        router = build_router(
            settings,
            FakeProvider("gemini", healthy=False),
            FakeProvider("groq", healthy=True),
        )

        statuses = {s.name: s for s in await router.refresh_health()}

        assert statuses["gemini"].healthy is False
        assert statuses["groq"].healthy is True

    async def test_failure_count_accumulates(self, settings: Settings) -> None:
        settings = Settings(
            gemini_api_key="k", groq_api_key="k", llm_health_ttl_seconds=0.0
        )
        router = build_router(settings, FakeProvider("gemini", fails=True), FakeProvider("groq"))

        await router.complete(MESSAGES)
        await router.complete(MESSAGES)

        gemini = next(s for s in router.status() if s.name == "gemini")
        assert gemini.consecutive_failures == 2
