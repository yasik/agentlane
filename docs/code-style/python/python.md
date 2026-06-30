# Python Conventions

These rules apply to Python code across the repository.

## Core conventions

- Target Python `3.12+`.
- Use strict typing with annotations for functions, methods, and classes.
- Follow Google-style formatting and Google-style docstrings for public
  functions, classes, and methods.
- Prefer explicit, readable code over overly clever or implicit code.
- Keep implementations simple and Pythonic.
- Separate logical control-flow blocks with blank lines when an early return,
  guard branch, or state transition finishes one idea and the next branch starts
  another. Do not pack adjacent `if`/`return` blocks together when spacing would
  make the flow easier to scan.
- Catch exceptions at the highest practical boundary for the operation, such as
  request, command-loop, run, worker, or I/O boundaries. Avoid catching broad
  exceptions inside low-level helpers unless the helper is itself that boundary;
  let actionable failures surface as typed errors or user-visible diagnostics.
- Do not model top-level domain, protocol, command, or response entities as
  generic `Mapping`/`dict` values that are passed between methods and read with
  `.get("field")`. Parse raw JSON or external dictionaries once at the trust
  boundary, then pass explicit dataclasses, Pydantic models, enums, or typed
  value objects through the rest of the code. Reserve dictionary access for
  true dynamic maps, caches, and low-level serialization helpers.
- Downstream adapters must not define their own string vocabulary for
  framework events. Expose supported event types from the upstream AgentLane
  package, then have bridges, transports, and apps consume those enums or
  constants so renames and additions are caught in one place. If multiple
  public enums need the same event name, define the literal in exactly one
  canonical enum and derive the other enum values from that source.
- For bridge, protocol, or transport dispatch, prefer explicit handler
  registries over long `isinstance` ladders. Each handler should declare the
  command or event type it handles, own that type's processing, and make its
  downstream side effects visible in the handler contract.

## Naming and structure

- Use snake case for variables and functions.
- Prefer descriptive names with auxiliary verbs where that improves clarity.
  Example: `is_active`
- Internal modules should use underscore-prefixed filenames. Detailed module and
  export rules live in [modules.md](./modules.md).

## Comments and docstrings

- Use [comments.md](./comments.md) for comment policy and examples.
- Dataclass and Pydantic fields must use this exact inline docstring style when
  a field comment is required:

```python
field: FieldType
"""Comment ..."""
```

- For Pydantic models used as schemas in LLM prompts, prefer
  `Field(description="...")`.
- For Pydantic models not used in prompts, prefer the inline docstring field
  style shown above.

## Functions and error handling

- Prefer functional style where it keeps logic clearer.
- Put validation and error handling near the start of the function.
- Use specific exception types and informative messages.
- Avoid bare `except`.
- Return `None` or an empty collection for "not found" cases instead of raising.
- Define error codes as enums when structured responses need them.

## Async and concurrency

- Prefer `async` and `await` for I/O-bound operations.
- Pass `CancellationToken` through async call chains for cancellation support.
- Use `asyncio.gather()` for concurrent operations when appropriate.
- Add explicit timeouts around external calls.

## Framework-specific conventions

- For FastAPI, use Pydantic models, clear return types, and explicit
  `HTTPException` responses.
- Use `structlog` for structured logging and include request IDs when available.

## Tooling and dependencies

- Use `uv` for dependency management. Avoid `pip` directly.
- Use `git` for version control and keep changes small and focused.
