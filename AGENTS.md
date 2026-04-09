# Agent Guidelines

## Philosophy

This codebase will outlive you. Every shortcut becomes someone else's burden. Every hack compounds into technical debt that slows the whole team down.

You are not just writing code. You are shaping the future of this project. The patterns you establish will be copied. The corners you cut will be cut again.

Fight entropy. Leave the codebase better than you found it.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `docs/plans/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- After code changes: spawn the `docs-keeper` agent to detect and update affected READMEs and tech specs

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `docs/plans/<task_name>.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `docs/plans/<task_name>.md`
6. **Capture Lessons**: Update `docs/plans/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Architecture

- Follow Single Responsibility Principle for modules
- Favor composition over inheritance
- Modular design with reusable components
- Nested packages for organization (shared dependencies at common level)

## Local workflow

1. Format, lint and type‑check your changes:

   ```bash
   /usr/bin/make format
   /usr/bin/make lint
   ```

2. Run the tests:

   ```bash
   /usr/bin/make tests
   ```

   To run a single test, use `uv run pytest -s -k <test_name>`.

All python commands should be run via `uv run python ...`

## Import Conventions

- Use package-root imports for public API usage (e.g., `from agentlane.messaging import AgentId`)
- Inside the same package, import private modules with relative imports only (e.g., `from ._identity import AgentId`)
- Never import private modules via full package paths (e.g., avoid `from agentlane.messaging._identity import AgentId`)
- Avoid wildcard imports (`from module import *`)
- Never use `from __future__ import annotations` unless explicitly approved for a specific file
- Never use `TYPE_CHECKING`
- **Never use imports inside functions or methods** - all imports must be at the top level of the module. Do not bypass this rule with `# pylint: disable` comments.

## Creating New Workspace Packages

To add a new utility package to the workspace:

### 1. Create Package Structure

```bash
# Create package directory with src layout
mkdir -p packages/<package-name>/src/agentlane_<package-name>

# Create required files
touch packages/<package-name>/src/agentlane_<package-name>/__init__.py
touch packages/<package-name>/src/agentlane_<package-name>/py.typed
```

### 2. Create `pyproject.toml`

Create `packages/<package-name>/pyproject.toml`:

```toml
[project]
name = "agentlane-<package-name>"
version = "0.1.0"
description = "Brief description of the package"
requires-python = ">=3.12"
dependencies = [
  # Add your dependencies here
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentlane_<package-name>"]

[tool.hatch.build.targets.wheel.sources]
"src" = ""
```

### 3. Register in Workspace

Update root `pyproject.toml`:

```toml
[project]
dependencies = [
  # ... existing dependencies
  "agentlane-<package-name>",
]

[tool.uv.sources]
agentlane-<package-name> = { workspace = true }
```

### 4. Install and Verify

```bash
# Sync workspace to install the new package
uv sync

# Verify it works
uv run python -c "import agentlane_<package-name>; print('✅ Package installed!')"
```

### Package Guidelines

- **Naming**: Use `agentlane_<name>` for imports, `agentlane-<name>` for package names
- **Structure**: Always use `src/` layout (`packages/<name>/src/agentlane_<name>/`)
- **Typing**: Include `py.typed` marker for type checking support
- **Tests**: Place tests in `<package>/tests/` or co-located with source
- **Dependencies**: Only declare direct dependencies in each package

## Code Style

- **Python Version:** 3.12+
- **Types:** Strict typing with annotations for all functions, methods, classes
- **Formatting:** Follow Google-style
- **Imports:** Package modules with underscore prefix (e.g., `_module.py`), export in `__init__.py` (see [Import Conventions](#import-conventions) section above)
- **Docstrings:** Use Google-style docstrings for all public methods, classes, and functions
- **Field Comments:** Dataclass and Pydantic fields must use this exact style:
  ```python
  field: FieldType
  """Comment ..."""
  ```
- **Comments:** Use inline comments sparingly; prefer docstrings for explanations on "why" rather than "what"
- **Testing:** Use `pytest` for unit tests, `pytest-asyncio` for async tests, `pytest-cov` for coverage
- **Test Fixtures:** Prefer defining fixtures in `conftest.py` files for better discoverability and reuse
- **Dependencies:** Use `uv` for dependency management, avoid `pip` directly
- **Version Control:** Use `git` for version control, follow conventional commits
- **Naming:** Snake case for variables/functions, descriptive names with auxiliary verbs (is_active)
- **File Structure:** Underscore-prefix for internals (`_module.py`), tests alongside code (`test_module.py`)
- **Functions:** Google-style docstrings, error handling at beginning, favor functional style
- **Error Handling:** Use specific exception types, provide informative messages, avoid bare excepts
- **Asynchronous:** Prefer async/await for I/O bound operations
- **FastAPI:** Use Pydantic models, clear return types, proper error responses with HTTPException
- **Pydantic Models:**
  - Use `Field(description="...")` when the model is used as a schema in LLM prompts (descriptions are included in the prompt)
  - Use inline docstring style for models not used in prompts:
    ```python
    field: str
    """Description of a field"""
    ```
- **Elegance and Readability:** Strive for elegant and Pythonic code that is easy to understand and maintain.
- **Explicit over Implicit:** Favor explicit code that clearly communicates its intent over implicit, overly concise code.
- **Zen of Python:** Keep the Zen of Python in mind when making design decisions.

## Additional Conventions

### File Naming

- `_types.py` for type definitions, `_agent.py` for agents, `_controller.py` for controllers
- `_errors.py` for custom exceptions, `_prompt.py` for LLM prompts

### Typing

- Use `strenum.LowercaseStrEnum` for string enumerations
- Use `TypeAlias` for complex type definitions
- Never use quoted string annotations for types (e.g., `"SomeType | None"`). Reorder class definitions or isolate shared types to avoid forward-reference hacks.

### Testing

- Name tests: `test_<what>_<condition>_<expected_result>`
- Mock external dependencies with `pytest-mock`
- **Pytest fixtures**: Always place fixtures in `conftest.py` files for better discoverability and reuse. Use `@pytest.fixture(name="<name>")` style to avoid pylint `redefined-outer-name` warnings. Name the fixture function as `fixture_<name>`:
  ```python
  # conftest.py
  @pytest.fixture(name="my_fixture")
  def fixture_my_fixture() -> MyType:
      return MyType()
  ```

### Async Patterns

- Pass `CancellationToken` through async call chains for cancellation support
- Use `asyncio.gather()` for concurrent operations; implement timeouts for external calls

### Error Handling

- Return `None` or empty collections for "not found" cases (don't raise)
- Define error codes as enums for structured responses

### Logging

- Use `structlog` for structured logging; include request IDs for tracing

## Modules Structure and Naming Patterns

1. Organize modules and sub-modules ensuring sub-packages are placed within packages that use them or that serve as an “umbrella” for them.
2. If a package is used by multiple others, place it at the level of those packages (think of least common denominator).
3. The decision between module vs subpackage is up to the developer and typically depends on the size of the code and its structure. The only thing to keep in mind is that if the module is public, it is very difficult to keep it clean in order to export only really related entities.

### \_\_init\_\_.py usage

1. The **\_\_init\_\_.py** file should solely focus on exporting entities to the external world.
2. Avoid declaring classes, types, or functions within **\_\_init\_\_.py**.
3. If any helpers have been imported in **\_\_init\_\_.py** to enable its logic, delete it in the end.

```py
import os

from ._events import DetectedEvent, StoredEvent
from ._spike import Spike

__all__ = [
    "DetectedEvent",
    "StoredEvent",
    "Spike",
]

DEVENV = False
if os.environ.get("DEVENV", False) == “True”:
    DEVENV = True

# delete helpers
del os
```

### Exports and internals

1. Prefix all modules and sub-packages intended for internal use with **\_**. Entities within such packages and modules should be either used internally only or exported within **\_\_init\_\_.py**.
2. Avoid re-exporting entities from public (not \_ prefixed packages and modules) within **\_\_init\_\_.py**. This leads to a confusion of what should be the proper import path for consumers of the package.
3. Export entities from the nearest nesting level only. Deeper levels should re-export on its own.

```py
# GOOD example of __init__.py
from ._events import DetectedEvent, StoredEvent
from ._spike import Spike

__all__ = [
    "DetectedEvent",
    "StoredEvent",
    "Spike",
]

# BAD example of __init__.py (confusing import paths)
# it is unclear what is correct usage for the consumer:
# - from pkg import Spike
# - from pkg.spike import Spike
from .events import DetectedEvent, StoredEvent
from .spike import Spike

__all__ = [
    "DetectedEvent",
    "StoredEvent",
    "Spike",
]

# BAD example of __init__.py (deep level re-export)
# in this case _events package has to export entities first
from ._events._detected import DetectedEvent
from ._events._stored import StoredEvent

__all__ = [
    "DetectedEvent",
    "StoredEvent",
]
```

```py
pkgname/
├── __init__.py
├── _module.py
├── _shared/            # used by "_internal_subpkg1" and "_internal_subpkg2"
├── _internal_subpkg1/  # uses shared as "from .._shared import ABC"
├── _internal_subpkg2/
└── public_subpkg1/
```

## Pull request expectations

- New tests are added when needed.
- Documentation is updated.
- `/usr/bin/make lint` and `/usr/bin/make format` have been run.
- The full test suite passes.
- Do not mention "Co-Authored" or "Authored By"

Commit messages should be concise and written in the imperative mood. Small, focused commits are preferred.
