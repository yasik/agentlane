# Code Commenting Best Practices

Thoughtful comments explain why code exists, how to use it, and capture non-obvious decisions; they do not restate what the code already says.

## Python specific:
- Use Google-style docstrings for public functions, classes, and methods.
- Dataclass and Pydantic field comments must use this exact style when inline
  field documentation is needed:

```python
field: FieldType
"""Comment ..."""
```

## Guiding principles

- Prefer clear code over comments; comments do not excuse poor naming or structure.  
- Focus comments on the **why** (intent, constraints, trade-offs), not the what (mechanics obvious from reading the code).  
- Keep comments accurate and maintained; outdated comments are worse than none.  
- Avoid noise: no comments for trivial operations or restating signatures.  
- Use consistent prefixes for work notes: `TODO:`, `FIXME:`, `HACK:`, `NOTE:`.  
- Let structured comments feed tools (Sphinx, Javadoc, Go doc, etc.) so documentation stays close to code.

A useful model:  
- Module/package comments: “What is this area of the system for?”  
- Class/type comments: “What abstraction does this represent?”  
- Function/method comments: “What does this do for callers?”  
- Inline comments: “Why is this line/branch written this way?”

***

## Module / package level comments

**Purpose:** Describe the module’s role, major concepts, invariants, and relationships to other modules.

**Best practices**

- Place a brief, high-level description at the top of the file.  
- Mention domain concepts, I/O and side effects, and important invariants or configuration expectations.  
- For libraries, include a tiny usage example when helpful.  

### Python module example (Google-style)

```python
"""User account management.

This module provides high-level operations for creating and managing user
accounts in the billing system. It coordinates persistence, email
notifications, and audit logging.

Typical usage example:

  account = create_account(email="user@example.com")
  enable_two_factor(account.id)
"""
```

### Go package example

```go
// Package accounts provides high-level operations for creating and
// managing user accounts in the billing system. It coordinates
// persistence, email notifications, and audit logging.
//
// Typical usage:
//
//   acc, err := accounts.Create(ctx, email)
//   if err != nil { /* handle error */ }
//   if err := accounts.EnableTwoFactor(ctx, acc.ID); err != nil { /* handle error */ }
package accounts
```

***

## Class / type level comments

**Purpose:** Describe what the abstraction represents, its invariants, and how it should be used.

**Best practices**

- Explain the conceptual model (“represents…”, “encapsulates…”), not just “Class that does X”.  
- Document invariants, lifecycle expectations, and thread-safety.  
- Call out extension points, required overrides, or important gotchas.  

### Python class example (Google-style)

```python
class RateLimiter:
  """Token-bucket rate limiter.

  This class enforces a maximum rate of events over time using a
  token-bucket algorithm. Tokens accumulate up to a fixed capacity
  and are consumed by calls to `acquire`.

  The implementation is thread-safe for concurrent callers.
  """

  def __init__(self, rate_per_second: float, capacity: int) -> None:
    self._rate_per_second = rate_per_second
    self._capacity = capacity
    # ...
```

### Go type example

```go
// RateLimiter enforces a maximum rate of events over time using a
// token-bucket algorithm. Tokens accumulate up to a fixed capacity
// and are consumed by calls to Acquire.
//
// RateLimiter is safe for concurrent use by multiple goroutines.
type RateLimiter struct {
    ratePerSecond float64
    capacity      int
    // ...
}
```

***

## Function / method level comments

**Purpose:** Specify contract, behavior, parameters, return values, errors, and important side effects.

**Best practices**

- For public/exposed APIs, always have a concise, complete doc comment.  
- Describe behavior at the level of the caller: what it does, not how it is implemented.  
- Include preconditions, postconditions, error conditions, and note blocking, I/O, or retries.  
- Internal helpers only need comments when intent or invariants are not obvious from naming.

### Python function examples (Google-style)

```python
def acquire(self, tokens: int, timeout_s: float) -> bool:
  """Attempt to acquire tokens from the rate limiter.

  This call may block up to `timeout_s` seconds while waiting for
  tokens to become available.

  Args:
    tokens: Number of tokens to acquire. Must be positive.
    timeout_s: Maximum time in seconds to wait for tokens.

  Returns:
    True if the tokens were acquired before the timeout, False otherwise.

  Raises:
    ValueError: If `tokens` is not positive.
  """
  # Implementation here
  ...
```

```python
from typing import Any

def parse_config(path: str) -> dict[str, Any]:
  """Load and validate application configuration from a file.

  The configuration is loaded from the given path and validated against
  the built-in schema. Relative paths are resolved relative to the
  current working directory.

  Args:
    path: Path to a JSON configuration file.

  Returns:
    Parsed configuration dictionary.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the configuration is invalid.
  """
  # Implementation here
  ...
```

### Go function examples

```go
// Acquire attempts to take tokens from the limiter.
// It may block for up to timeout while waiting for tokens.
//
// It returns true if the tokens were acquired before the timeout.
func (l *RateLimiter) Acquire(ctx context.Context, tokens int, timeout time.Duration) bool {
    // Implementation here
    return false
}
```

```go
// ParseConfig loads application configuration from path and validates
// it against the built-in schema.
//
// It returns an error if the file cannot be read or the configuration
// is invalid.
func ParseConfig(path string) (*Config, error) {
    // Implementation here
    return nil, nil
}
```

***

## Inline and block comments inside functions

**Purpose:** Clarify non-obvious logic, document invariants, and highlight rationale for unusual code.

**Best practices**

- Prefer full-line comments above the code they explain.  
- Reserve end-of-line comments for very short, local clarifications.  
- Explain why something is done, or which external constraint it satisfies (spec requirement, bug workaround, performance trade-off).  
- Document tricky invariants, assumptions about inputs, or non-obvious algorithm steps.  
- Do not narrate each line or describe obvious control flow.

### Python inline comment examples

```python
def _rebuild_index(entries: list[str]) -> dict[str, int]:
  """Builds an index of entry positions.

  Args:
    entries: Sequence of unique entry IDs.

  Returns:
    Mapping from entry ID to position in `entries`.
  """
  index: dict[str, int] = {}

  # Rely on enumerate order to define the canonical position.
  # This is O(n), but called only at startup, so it's acceptable.
  for pos, entry_id in enumerate(entries):
    index[entry_id] = pos

  return index
```

```python
def send_with_retries(message: bytes, max_retries: int) -> None:
  """Send a message to the remote endpoint with bounded retries."""
  attempt = 0

  while attempt <= max_retries:
    try:
      # We intentionally do not retry inside send_once; higher-level
      # retry with backoff is centralized here.
      send_once(message)
      return
    except TransientError:
      attempt += 1

  raise RuntimeError("exhausted retries")
```

### Go inline comment examples

```go
func rebuildIndex(entries []string) map[string]int {
    index := make(map[string]int, len(entries))

    // Use the slice order as the canonical position. This is O(n) but
    // happens only at startup.
    for pos, id := range entries {
        index[id] = pos
    }
    return index
}
```

```go
func SendWithRetries(ctx context.Context, msg []byte, maxRetries int) error {
    attempt := 0

    for attempt <= maxRetries {
        // Do not retry inside sendOnce; all retry logic (including
        // backoff) lives here to centralize behavior.
        if err := sendOnce(ctx, msg); err == nil {
            return nil
        }
        attempt++
    }

    return fmt.Errorf("exhausted retries after %d attempts", maxRetries)
}
```

***

## “Bad comment” patterns to avoid

- Restating code: `// increment i` above `i++`.  
- Commented-out code left in the repo instead of relying on version control.  
- Stale comments that contradict actual behavior.  
- Overly long “essays” that obscure code structure.  
- Jokes, sarcasm, or unclear metaphors in core logic.  

Rule of thumb: if deleting a comment would not remove useful knowledge for a future maintainer, that comment probably should not exist.

***

## Patterns summary

| Area              | Common pattern in good repos                                                |
|-------------------|-------------------------------------------------------------------------------|
| Public APIs       | Every exported symbol has a concise, accurate doc comment.                  |
| Packages/modules  | Top-of-file comment describing purpose, often with a tiny usage example.    |
| Internal helpers  | Emphasis on good naming; comments only for tricky logic or invariants.      |
| Invariants        | Explicitly documented near the code that enforces them.                     |
| TODOs / FIXMEs    | Tagged with owner/issue and short context, not used as a dumping ground.    |
