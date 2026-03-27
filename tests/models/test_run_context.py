"""Tests for DefaultRunContext thread-safe atomic operations."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from agentlane.models.run import DefaultRunContext, RunContext

def run_async[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an awaitable inside a fresh event loop for sync pytest tests."""
    return asyncio.run(awaitable)


class TestRunContext:
    """Tests for the base RunContext class."""

    def test_stores_context(self) -> None:
        """RunContext should store the provided context object."""
        context = {"key": "value"}
        run_context = RunContext(context=context)
        assert run_context.context == context

    def test_context_is_mutable(self) -> None:
        """RunContext should expose the original mutable context object."""
        context: dict[str, str] = {}
        run_context = RunContext(context=context)
        run_context.context["key"] = "value"
        assert run_context.context["key"] == "value"


class TestDefaultRunContextBasics:
    """Tests for the DefaultRunContext MutableMapping interface."""

    def test_default_empty_context(self) -> None:
        """DefaultRunContext should start with an empty dictionary."""
        run_context = DefaultRunContext()
        assert run_context.context == {}
        assert len(run_context) == 0

    def test_getitem_setitem(self) -> None:
        """Dict-like item access should work."""
        run_context = DefaultRunContext()
        run_context["key"] = "value"
        assert run_context["key"] == "value"

    def test_delitem(self) -> None:
        """Deleting an item should remove it from the context."""
        run_context = DefaultRunContext()
        run_context["key"] = "value"
        del run_context["key"]
        assert "key" not in run_context

    def test_iter(self) -> None:
        """Iteration should yield the stored keys."""
        run_context = DefaultRunContext()
        run_context["a"] = 1
        run_context["b"] = 2
        assert set(run_context) == {"a", "b"}

    def test_len(self) -> None:
        """Length should reflect the number of stored items."""
        run_context = DefaultRunContext()
        assert len(run_context) == 0
        run_context["a"] = 1
        run_context["b"] = 2
        assert len(run_context) == 2

    def test_clear(self) -> None:
        """Clear should remove all items from the context."""
        run_context = DefaultRunContext()
        run_context["a"] = 1
        run_context["b"] = 2
        run_context.clear()
        assert len(run_context) == 0


class TestDefaultRunContextIncrement:
    """Tests for the increment atomic operation."""

    def test_increment_creates_key_with_default(self) -> None:
        """Increment should create a missing key using the default value."""
        async def exercise() -> tuple[int, DefaultRunContext]:
            run_context = DefaultRunContext()
            result = await run_context.increment("counter")
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == 1
        assert run_context["counter"] == 1

    def test_increment_with_custom_default(self) -> None:
        """Increment should honor a custom default."""
        async def exercise() -> tuple[int, DefaultRunContext]:
            run_context = DefaultRunContext()
            result = await run_context.increment("counter", default=10)
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == 11
        assert run_context["counter"] == 11

    def test_increment_existing_value(self) -> None:
        """Increment should update an existing counter."""
        async def exercise() -> tuple[int, DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["counter"] = 5
            result = await run_context.increment("counter")
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == 6
        assert run_context["counter"] == 6

    def test_increment_multiple_times(self) -> None:
        """Multiple sequential increments should produce increasing values."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            for i in range(1, 6):
                result = await run_context.increment("counter")
                assert result == i
            return run_context

        run_context = run_async(exercise())
        assert run_context["counter"] == 5

    def test_increment_concurrent_access(self) -> None:
        """Concurrent increments should remain atomic."""
        async def exercise() -> tuple[list[int], DefaultRunContext]:
            run_context = DefaultRunContext()
            num_tasks = 100

            async def increment_task() -> int:
                return await run_context.increment("counter")

            results = await asyncio.gather(
                *[increment_task() for _ in range(num_tasks)]
            )
            return results, run_context

        results, run_context = run_async(exercise())
        num_tasks = 100

        assert sorted(results) == list(range(1, num_tasks + 1))
        assert run_context["counter"] == num_tasks


class TestDefaultRunContextSet:
    """Tests for the set atomic operation."""

    def test_set_creates_key(self) -> None:
        """Set should create a new key when missing."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            await run_context.set("key", "value")
            return run_context

        run_context = run_async(exercise())
        assert run_context["key"] == "value"

    def test_set_overwrites_existing(self) -> None:
        """Set should overwrite an existing value."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            run_context["key"] = "old"
            await run_context.set("key", "new")
            return run_context

        run_context = run_async(exercise())
        assert run_context["key"] == "new"

    def test_set_concurrent_access(self) -> None:
        """Concurrent set operations should not corrupt state."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            num_tasks = 50

            async def set_task(value: int) -> None:
                await run_context.set("key", value)

            await asyncio.gather(*[set_task(i) for i in range(num_tasks)])
            return run_context

        run_context = run_async(exercise())
        num_tasks = 50

        assert run_context["key"] in range(num_tasks)


class TestDefaultRunContextAppendToList:
    """Tests for the append_to_list atomic operation."""

    def test_append_creates_list_if_missing(self) -> None:
        """append_to_list should create a list when the key is missing."""
        async def exercise() -> tuple[list[object], DefaultRunContext]:
            run_context = DefaultRunContext()
            result = await run_context.append_to_list("items", "first")
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == ["first"]
        assert run_context["items"] == ["first"]

    def test_append_to_existing_list(self) -> None:
        """append_to_list should append to an existing list."""
        async def exercise() -> tuple[list[object], DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["items"] = ["existing"]
            result = await run_context.append_to_list("items", "new")
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == ["existing", "new"]
        assert run_context["items"] == ["existing", "new"]

    def test_append_multiple_items(self) -> None:
        """Sequential append_to_list calls should preserve order."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            for i in range(5):
                await run_context.append_to_list("items", i)
            return run_context

        run_context = run_async(exercise())
        assert run_context["items"] == [0, 1, 2, 3, 4]

    def test_append_concurrent_access(self) -> None:
        """Concurrent appends should preserve all items."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            num_tasks = 100

            async def append_task(value: int) -> None:
                await run_context.append_to_list("items", value)

            await asyncio.gather(*[append_task(i) for i in range(num_tasks)])
            return run_context

        run_context = run_async(exercise())
        num_tasks = 100

        assert sorted(run_context["items"]) == list(range(num_tasks))
        assert len(run_context["items"]) == num_tasks


class TestDefaultRunContextExtendList:
    """Tests for the extend_list atomic operation."""

    def test_extend_creates_list_if_missing(self) -> None:
        """extend_list should create a list when the key is missing."""
        async def exercise() -> tuple[list[object], DefaultRunContext]:
            run_context = DefaultRunContext()
            result = await run_context.extend_list("items", [1, 2, 3])
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == [1, 2, 3]
        assert run_context["items"] == [1, 2, 3]

    def test_extend_existing_list(self) -> None:
        """extend_list should append multiple values to an existing list."""
        async def exercise() -> tuple[list[object], DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["items"] = [1, 2]
            result = await run_context.extend_list("items", [3, 4])
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == [1, 2, 3, 4]
        assert run_context["items"] == [1, 2, 3, 4]

    def test_extend_with_empty_list(self) -> None:
        """Extending by an empty list should leave the original list intact."""
        async def exercise() -> tuple[list[object], DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["items"] = [1, 2]
            result = await run_context.extend_list("items", [])
            return result, run_context

        result, run_context = run_async(exercise())
        assert result == [1, 2]
        assert run_context["items"] == [1, 2]

    def test_extend_concurrent_access(self) -> None:
        """Concurrent extend_list calls should preserve all values."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()
            num_tasks = 50

            async def extend_task(start: int) -> None:
                await run_context.extend_list("items", [start, start + 1])

            await asyncio.gather(*[extend_task(i * 2) for i in range(num_tasks)])
            return run_context

        run_context = run_async(exercise())
        num_tasks = 50

        expected = list(range(num_tasks * 2))
        assert sorted(run_context["items"]) == expected
        assert len(run_context["items"]) == num_tasks * 2


class TestDefaultRunContextAppendIfUnique:
    """Tests for the append_if_unique atomic operation."""

    def test_append_if_unique_creates_list(self) -> None:
        """append_if_unique should create a new list when missing."""
        async def exercise() -> tuple[bool, DefaultRunContext]:
            run_context = DefaultRunContext()
            result = await run_context.append_if_unique(
                "items",
                "first",
                key_fn=lambda x: x,
            )
            return result, run_context

        result, run_context = run_async(exercise())
        assert result is True
        assert run_context["items"] == ["first"]

    def test_append_if_unique_adds_unique(self) -> None:
        """append_if_unique should append unique values."""
        async def exercise() -> tuple[bool, DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["items"] = ["a", "b"]
            result = await run_context.append_if_unique(
                "items",
                "c",
                key_fn=lambda x: x,
            )
            return result, run_context

        result, run_context = run_async(exercise())
        assert result is True
        assert run_context["items"] == ["a", "b", "c"]

    def test_append_if_unique_rejects_duplicate(self) -> None:
        """append_if_unique should reject duplicate values."""
        async def exercise() -> tuple[bool, DefaultRunContext]:
            run_context = DefaultRunContext()
            run_context["items"] = ["a", "b"]
            result = await run_context.append_if_unique(
                "items",
                "a",
                key_fn=lambda x: x,
            )
            return result, run_context

        result, run_context = run_async(exercise())
        assert result is False
        assert run_context["items"] == ["a", "b"]

    def test_append_if_unique_with_custom_key_fn(self) -> None:
        """append_if_unique should use the provided uniqueness function."""
        item1 = {"id": 1, "name": "first"}
        item2 = {"id": 2, "name": "second"}
        item3 = {"id": 1, "name": "duplicate"}

        async def exercise() -> tuple[bool, DefaultRunContext]:
            run_context = DefaultRunContext()
            await run_context.append_if_unique(
                "items",
                item1,
                key_fn=lambda x: x["id"],
            )
            await run_context.append_if_unique(
                "items",
                item2,
                key_fn=lambda x: x["id"],
            )
            result = await run_context.append_if_unique(
                "items",
                item3,
                key_fn=lambda x: x["id"],
            )
            return result, run_context

        result, run_context = run_async(exercise())

        assert result is False
        assert len(run_context["items"]) == 2
        assert run_context["items"] == [item1, item2]

    def test_append_if_unique_concurrent_access(self) -> None:
        """Concurrent append_if_unique calls should remain atomic."""
        async def exercise() -> tuple[list[bool], DefaultRunContext]:
            run_context = DefaultRunContext()
            num_tasks = 100

            async def append_task(value: int) -> bool:
                return await run_context.append_if_unique(
                    "items",
                    value % 50,
                    key_fn=lambda x: x,
                )

            results = await asyncio.gather(*[append_task(i) for i in range(num_tasks)])
            return results, run_context

        results, run_context = run_async(exercise())

        assert sum(results) == 50
        assert len(run_context["items"]) == 50
        assert sorted(run_context["items"]) == list(range(50))


class TestDefaultRunContextConcurrentMixedOperations:
    """Tests for mixed concurrent operations on DefaultRunContext."""

    def test_mixed_operations_concurrent(self) -> None:
        """Different atomic operations should remain safe under concurrency."""
        async def exercise() -> DefaultRunContext:
            run_context = DefaultRunContext()

            async def increment_task() -> None:
                for _ in range(10):
                    await run_context.increment("counter")

            async def append_task() -> None:
                for i in range(10):
                    await run_context.append_to_list("items", i)

            async def set_task() -> None:
                for i in range(10):
                    await run_context.set("value", i)

            await asyncio.gather(
                increment_task(),
                increment_task(),
                append_task(),
                append_task(),
                set_task(),
            )
            return run_context

        run_context = run_async(exercise())

        assert run_context["counter"] == 20
        assert len(run_context["items"]) == 20
        assert run_context["value"] in range(10)
