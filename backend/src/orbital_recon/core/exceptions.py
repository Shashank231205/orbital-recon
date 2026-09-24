"""Domain exceptions.

Each carries an HTTP status so the API layer can translate without inspecting
exception types case by case.
"""

from http import HTTPStatus


class OrbitalReconError(Exception):
    """Base class for all application errors."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(OrbitalReconError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "validation_error"


class NotFoundError(OrbitalReconError):
    status_code = HTTPStatus.NOT_FOUND
    code = "not_found"


class UnsupportedMediaError(OrbitalReconError):
    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media"


class PayloadTooLargeError(OrbitalReconError):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    code = "payload_too_large"


class ModelLoadError(OrbitalReconError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "model_unavailable"


class InferenceError(OrbitalReconError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "inference_failed"


class LLMProviderError(OrbitalReconError):
    """A single provider failed; the router may still fall back to another."""

    status_code = HTTPStatus.BAD_GATEWAY
    code = "llm_provider_error"


class AllProvidersUnavailableError(OrbitalReconError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "llm_unavailable"
