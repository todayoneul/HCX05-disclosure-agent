"""Stable, secret-free HCX transport error types."""

from __future__ import annotations


class HcxError(RuntimeError):
    """Base class for every production HCX failure."""


class HcxConfigurationError(HcxError):
    """Local runtime configuration is missing or invalid."""


class HcxContractError(HcxError):
    """A request violates the local native-v3 contract."""


class HcxTransportError(HcxError):
    """The remote service could not be reached."""

    retryable = True


class HcxConnectTimeout(HcxTransportError):
    """The configured connection deadline expired."""


class HcxReadTimeout(HcxTransportError):
    """The configured response-read deadline expired."""


class HcxHttpError(HcxError):
    """The service returned a non-success HTTP status."""

    retryable = False

    def __init__(self, http_status: int) -> None:
        self.http_status = http_status
        super().__init__(f"HCX HTTP request failed with status {http_status}")


class HcxRateLimitError(HcxHttpError):
    """The service rate-limited a request."""

    retryable = True

    def __init__(self, *, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(429)


class HcxServerError(HcxHttpError):
    """The service returned a retryable 5xx status."""

    retryable = True


class HcxApiError(HcxError):
    """HTTP succeeded but the HCX status contract reported a failure."""

    retryable = False

    def __init__(self, api_code: str) -> None:
        self.api_code = api_code
        super().__init__(f"HCX API request failed with code {api_code}")


class HcxResponseError(HcxError):
    """The remote payload does not satisfy the native-v3 response contract."""
