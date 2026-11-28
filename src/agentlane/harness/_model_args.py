"""Copy request data without copying live model or transport resources."""

import copy
from collections.abc import Iterable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any, cast

from agentlane.models import Model


def snapshot_model_args(
    arguments: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Own argument data while retaining model and context-manager handles.

    Resource handles inside argument containers remain shared dependencies.
    Other values use normal deep-copy semantics; unsupported data still raises
    instead of silently sharing a mutable value.
    """
    if arguments is None:
        return None

    # Seed the copy memo with resources before copying the surrounding data.
    # A visited set keeps shared or cyclic argument containers safe to inspect.
    memo: dict[int, Any] = {}
    _retain_resources(arguments, memo, set())

    return copy.deepcopy(arguments, memo)


def _retain_resources(value: object, memo: dict[int, Any], seen: set[int]) -> None:
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if isinstance(value, (Model, AbstractContextManager, AbstractAsyncContextManager)):
        memo[identity] = value
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            _retain_resources(key, memo, seen)
            _retain_resources(item, memo, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in cast(Iterable[object], value):
            _retain_resources(item, memo, seen)
