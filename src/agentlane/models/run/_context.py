import asyncio
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

TContext = TypeVar("TContext", default=Any)
T = TypeVar("T")


@dataclass
class RunContext(Generic[TContext]):
    """The context for the agent run.

    NOTE: Contexts are not passed to the LLM. They're a way to pass dependencies and
    data to code you implement, like tool functions, callabacks, hooks, etc.

    A very common use is to pass the in-between results to the next tool call if
    the previous tool call result is sufficiently complex object to be passed as
    an argument to the next tool call.
    """

    context: TContext
    """The context object (or None) for the agent run."""


@dataclass
class DefaultRunContext(RunContext[dict[str, Any]], MutableMapping[str, Any]):
    """Dictionary-backed run context for sharing state between tool calls.

    Thread-safe for concurrent asyncio access via atomic helper methods.

    Attributes:
        context: Mutable mapping that stores intermediate results for a single
            agent invocation.
    """

    context: dict[str, Any] = field(default_factory=lambda: {})
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.context[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.context[key] = value

    def __delitem__(self, key: str) -> None:
        del self.context[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.context)

    def __len__(self) -> int:
        return len(self.context)

    def clear(self) -> None:
        """Remove all items stored in the context."""

        self.context.clear()

    async def increment(self, key: str, default: int = 0) -> int:
        """Atomically increment a counter and return the new value.

        Args:
            key: The key for the counter.
            default: The default value if the key doesn't exist.

        Returns:
            The new value after incrementing.
        """
        async with self._lock:
            value = self.context.get(key, default) + 1
            self.context[key] = value
            return value

    async def set(self, key: str, value: Any) -> None:
        """Atomically set a key's value.

        Args:
            key: The key to set.
            value: The value to set.
        """
        async with self._lock:
            self.context[key] = value

    async def append_to_list(self, key: str, value: Any) -> list[Any]:
        """Atomically append a value to a list, creating it if needed.

        Args:
            key: The key for the list.
            value: The value to append.

        Returns:
            The list after appending.
        """
        async with self._lock:
            if key not in self.context:
                self.context[key] = []
            self.context[key].append(value)
            return self.context[key]

    async def extend_list(self, key: str, values: list[Any]) -> list[Any]:
        """Atomically extend a list with multiple values.

        Args:
            key: The key for the list.
            values: The values to extend with.

        Returns:
            The list after extending.
        """
        async with self._lock:
            if key not in self.context:
                self.context[key] = []
            self.context[key].extend(values)
            return self.context[key]

    async def append_if_unique(
        self,
        key: str,
        value: T,
        key_fn: Callable[[T], Any],
    ) -> bool:
        """Atomically append if value is unique based on key function.

        Args:
            key: The key for the list.
            value: The value to append.
            key_fn: Function to extract the unique key from items.

        Returns:
            True if the value was appended, False if it was a duplicate.
        """
        async with self._lock:
            if key not in self.context:
                self.context[key] = []
            existing_keys = {key_fn(item) for item in self.context[key]}
            if key_fn(value) not in existing_keys:
                self.context[key].append(value)
                return True
            return False
