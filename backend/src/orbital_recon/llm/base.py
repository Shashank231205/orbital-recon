"""Provider-agnostic interface for language model access.

The system must keep answering when a provider rate-limits or goes down, so no
call site depends on a specific vendor. Providers implement this interface and
the router decides which one serves a given request.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in a conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A model response, annotated with the provider that produced it.

    The provider name is surfaced to the client so an operator can see which
    backend answered, which matters when one has degraded.
    """

    text: str
    provider: str
    model: str
    latency_ms: int


class LLMProvider(ABC):
    """A language model backend."""

    name: str

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether credentials are present for this provider."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Generate a completion.

        Raises:
            LLMProviderError: If the provider fails or returns an unusable response.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Whether the provider is currently able to serve requests."""

    async def aclose(self) -> None:
        """Release any held connections.

        Optional: providers holding no resources need not override this.
        """
        return None
