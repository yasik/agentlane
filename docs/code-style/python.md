# Python Conventions

These rules apply to Python code across the repository.

## Core conventions

- Target Python `3.12+`.
- Use strict typing with annotations for functions, methods, and classes.
- Follow Google-style formatting and Google-style docstrings for public
  functions, classes, and methods.
- Prefer explicit, readable code over overly clever or implicit code.
- Keep implementations simple and Pythonic.

## Naming and structure

- Use snake case for variables and functions.
- Prefer descriptive names with auxiliary verbs where that improves clarity.
  Example: `is_active`
- Internal modules should use underscore-prefixed filenames. Detailed module and
  export rules live in [modules.md](./modules.md).

## Statement grouping

- Separate logical blocks of code with a single blank line rather than packing
  distinct steps into an unbroken run. Successive guard clauses with different
  intent each read as their own block:

```python
if allowed is None:
    return INHERIT_TOOLS, shims

if not allowed:
    return OVERRIDE_TOOLS, shims

return RESTRICT_TOOLS.only(*allowed), shims
```

- Add a short inline comment before a block only where it clarifies intent; do
  not narrate self-evident code.

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
- Return `None` or an empty collection for "not found" cases instead of raising.
- Do not catch exceptions too early or too deep. Let a system- or user-fixable
  error (for example a missing or unreadable file from `read_text`) propagate to
  a layer that can act on it, instead of swallowing it low in the stack where it
  becomes silent and hard to debug.
- When a soft limit applies, prefer visible truncation with a pointer to the
  full source (for example the file path) over silently dropping data, so the
  consumer can recover the rest.
- Do not add logging as a substitute for surfacing an error; prefer raising or
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
