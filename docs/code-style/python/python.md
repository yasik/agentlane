# Python Conventions

These rules apply to Python code across the repository.

## Core conventions

- Target Python `3.12+`.
- Use strict typing with annotations for functions, methods, and classes.
- Follow Google-style formatting and Google-style docstrings for public
  functions, classes, and methods.
- Prefer explicit, readable code over overly clever or implicit code.
- Keep implementations simple and Pythonic.
- Do not reduce line count at the expense of readability, type enforcement, or
  extension clarity. A simplification should remove real duplication or make an
  invariant more explicit; it should not flatten a useful abstraction into a
  looser one.
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
- Keep class-based extension seams when they encode real obligations such as
  accepted input type, side-effect ownership, emitted event types, lifecycle
  state, or override points. Do not replace those classes with loose functions
  merely to reduce boilerplate.
- Keep representative fixtures and parity tests close to small hand-authored
  protocols. Do not introduce generated manifests or runtime discovery files
  unless the repository genuinely generates or discovers protocol definitions
  from them.

## Naming and structure

- Use snake case for variables and functions.
- Prefer descriptive names with auxiliary verbs where that improves clarity.
  Example: `is_active`
- Internal modules should use underscore-prefixed filenames. Detailed module and
  export rules live in [modules.md](./modules.md).

## Comments and docstrings

- Use [comments.md](./comments.md) for comment policy and examples.
- In docstrings and comments, use single backticks for inline code and
  identifiers (`model_args`), not double backticks.
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
- Avoid catching broad exceptions deep in helpers. Low-level code should raise
  typed, actionable errors and let the request, run, command-loop, worker, or
  I/O boundary translate those errors into user-visible diagnostics.
- Return `None` or an empty collection for "not found" cases instead of raising.
- When a soft limit applies, prefer visible truncation with a pointer to the
  full source, such as the file path, over silently dropping data so the
  consumer can recover the rest.
- Do not add logging as a substitute for surfacing an error. Prefer raising or
  returning a clear value over logging-and-continuing.
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
