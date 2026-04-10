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
