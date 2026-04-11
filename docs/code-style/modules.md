# Module Layout and Exports

Use this page for file naming, package layout, `__init__.py`, and export rules.

## File naming

- Use `_types.py` for type definitions.
- Use `_agent.py` for agents.
- Use `_controller.py` for controllers.
- Use `_errors.py` for custom exceptions.
- Use `_prompt.py` for LLM prompts.

## Package and module structure

1. Organize modules and subpackages so related code stays together.
2. If a package is used by multiple others, place it at their least common
   level.
3. Choose between a module and a subpackage based on the size and cohesion of
   the code. Public surfaces should stay easy to understand and export cleanly.

Example layout:

```py
pkgname/
├── __init__.py
├── _module.py
├── _shared/
├── _internal_subpkg1/
├── _internal_subpkg2/
└── public_subpkg1/
```

## `__init__.py` usage

1. Use `__init__.py` only to export public entities.
2. Do not declare classes, functions, or types in `__init__.py`.
3. If `__init__.py` imports helpers for its own logic, delete those helpers at
   the end of the file.

Example:

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
if os.environ.get("DEVENV", False) == "True":
    DEVENV = True

del os
```

## Exports and internals

1. Prefix internal modules and subpackages with `_`.
2. Entities inside internal modules should stay internal unless exported from
   the nearest `__init__.py`.
3. Avoid re-exporting entities from public modules in `__init__.py`; that makes
   import paths unclear for consumers.
4. Export entities from the nearest nesting level only. Deeper packages should
   re-export their own public surface.

Examples:

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
from .events import DetectedEvent, StoredEvent
from .spike import Spike

__all__ = [
    "DetectedEvent",
    "StoredEvent",
    "Spike",
]

# BAD example of __init__.py (deep level re-export)
from ._events._detected import DetectedEvent
from ._events._stored import StoredEvent

__all__ = [
    "DetectedEvent",
    "StoredEvent",
]
```
