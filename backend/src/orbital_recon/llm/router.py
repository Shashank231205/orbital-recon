"""Failover routing across language model providers.

Providers are tried in configured order. A provider that fails is marked
unhealthy for a cooldown window so subsequent requests skip it immediately
instead of paying its timeout again, and is retried once the window expires.
This keeps a single degraded vendor from slowing every request.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from orbital_recon.core.config import Settings
from orbital_recon.core.exceptions import AllProvidersUnavailableError, LLMProviderError
from orbital_recon.core.logging import get_logger
from orbital_recon.llm.base import ChatMessage, CompletionResult, LLMProvider
from orbital_recon.llm.providers.gemini import GeminiProvider
from orbital_recon.llm.providers.groq import GroqProvider

logger = get_logger(__name__)


@dataclass
class _ProviderState:
    """Health bookkeeping for one provider."""

    healthy: bool = True
    checked_at: float = 0.0
    consecutive_failures: int = 0


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Provider health as reported to operators."""

    name: str
    configured: bool
    healthy: bool
    consecutive_failures: int


class LLMRouter:
    """Serves completions from the first healthy provider."""

    def __init__(
        self,
        settings: Settings,
        providers: Sequence[LLMProvider] | None = None,
    ) -> None:
        """Build a router.

        Args:
            settings: Supplies credentials, ordering and cooldown.
            providers: Explicit providers, overriding those built from
                settings. Used by tests and by deployments that construct
                providers themselves.
        """
        self._cooldown = settings.llm_health_ttl_seconds

        if providers is not None:
            self._providers = list(providers)
            self._state = {p.name: _ProviderState() for p in self._providers}
            return

        available: dict[str, LLMProvider] = {
            "gemini": GeminiProvider(
                settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds
            ),
            "groq": GroqProvider(
                settings.groq_api_key, settings.groq_model, settings.llm_timeout_seconds
            ),
        }

        # Configured order determines priority; unknown names are ignored so a
        # typo degrades to fewer providers rather than crashing at startup.
        self._providers: list[LLMProvider] = [
            available[name] for name in settings.llm_provider_order if name in available
        ]
        self._state: dict[str, _ProviderState] = {
            provider.name: _ProviderState() for provider in self._providers
        }

    @property
    def has_configured_provider(self) -> bool:
        return any(provider.is_configured for provider in self._providers)

    def _is_available(self, provider: LLMProvider) -> bool:
        """Whether to attempt this provider now."""
        if not provider.is_configured:
            return False

        state = self._state[provider.name]
        if state.healthy:
            return True

        # Cooldown elapsed: allow one probe so a recovered provider comes back.
        return (time.monotonic() - state.checked_at) >= self._cooldown

    def _record_success(self, provider: LLMProvider) -> None:
        state = self._state[provider.name]
        state.healthy = True
        state.checked_at = time.monotonic()
        state.consecutive_failures = 0

    def _record_failure(self, provider: LLMProvider) -> None:
        state = self._state[provider.name]
        state.healthy = False
        state.checked_at = time.monotonic()
        state.consecutive_failures += 1

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Return a completion from the first provider that succeeds.

        Raises:
            AllProvidersUnavailableError: If every provider is unconfigured,
                cooling down, or failing.
        """
        attempted: list[str] = []

        for provider in self._providers:
            if not self._is_available(provider):
                continue

            attempted.append(provider.name)
            try:
                result = await provider.complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            except LLMProviderError as exc:
                self._record_failure(provider)
                logger.warning(
                    "llm_provider_failed", provider=provider.name, error=str(exc)
                )
                continue

            self._record_success(provider)
            logger.info(
                "llm_completion", provider=provider.name, latency_ms=result.latency_ms
            )
            return result

        raise AllProvidersUnavailableError(
            "No language model provider could serve the request",
            details={"attempted": attempted},
        )

    async def refresh_health(self) -> list[ProviderStatus]:
        """Probe every configured provider and return current status."""
        statuses: list[ProviderStatus] = []

        for provider in self._providers:
            configured = provider.is_configured
            healthy = await provider.health_check() if configured else False

            state = self._state[provider.name]
            state.healthy = healthy
            state.checked_at = time.monotonic()
            if healthy:
                state.consecutive_failures = 0

            statuses.append(
                ProviderStatus(
                    name=provider.name,
                    configured=configured,
                    healthy=healthy,
                    consecutive_failures=state.consecutive_failures,
                )
            )

        return statuses

    def status(self) -> list[ProviderStatus]:
        """Current cached status without issuing network calls."""
        return [
            ProviderStatus(
                name=provider.name,
                configured=provider.is_configured,
                healthy=self._state[provider.name].healthy,
                consecutive_failures=self._state[provider.name].consecutive_failures,
            )
            for provider in self._providers
        ]

    async def aclose(self) -> None:
        for provider in self._providers:
            await provider.aclose()
