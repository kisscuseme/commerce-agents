from __future__ import annotations


class ModelRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.provider_request_id = provider_request_id


class AuthenticationError(ModelRuntimeError):
    pass


class RateLimitError(ModelRuntimeError):
    pass


class TransientProviderError(ModelRuntimeError):
    pass


class ModelUnavailableError(ModelRuntimeError):
    pass


class InvalidRequestError(ModelRuntimeError):
    pass


class UnsupportedCapabilityError(ModelRuntimeError):
    pass


class ProviderProtocolError(ModelRuntimeError, ValueError):
    """Malformed provider/client protocol data; also a ValueError for v1 compatibility."""


class StreamInterruptedError(ModelRuntimeError):
    pass


class MalformedToolCall(ValueError):
    def __init__(self, message: str, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
