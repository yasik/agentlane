"""Model resolution seam for markdown agent definitions.

A frontmatter `model` is a provider/model spec string (for example
`anthropic/claude-...`). Turning it into a live `Model` client needs
credentials, which are deliberately not in the file. So core ships only a
`ModelResolver` protocol plus a thin adapter over a caller-supplied `Factory`;
it never imports the optional provider packages or reads environment variables.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from agentlane.models import Factory, Model, ModelResponse


class ModelResolver(Protocol):
    """Build a live model client from a provider/model spec string."""

    def resolve(
        self,
        model_spec: str,
        *,
        model_args: dict[str, Any],
    ) -> Model[ModelResponse]:
        """Return a client for `model_spec`.

        `model_args` is provided for context only. Per-call arguments are
        carried on the descriptor's `model_args` (forwarded by the runner as
        `extra_call_args`) and should not be baked into the client here.
        """
        ...


@dataclass(frozen=True, slots=True)
class FactoryModelResolver:
    """Resolve a model spec by routing it through a pre-built `Factory`.

    The factory carries credentials and provider routing; the spec string is its
    routing key. `model_args` is intentionally ignored here — it lives on the
    descriptor as a single source of truth and is applied per call.
    """

    factory: Factory[ModelResponse]
    """Caller-supplied factory that builds clients and owns credentials."""

    def resolve(
        self,
        model_spec: str,
        *,
        model_args: dict[str, Any],
    ) -> Model[ModelResponse]:
        del model_args
        return self.factory.get_model_client(model=model_spec)
