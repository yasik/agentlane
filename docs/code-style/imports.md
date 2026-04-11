# Imports and Typing

Use these rules for imports, annotations, and type definitions.

## Import conventions

- Use package-root imports for public API usage.
  Example: `from agentlane.messaging import AgentId`
- Inside the same package, import private modules with relative imports only.
  Example: `from ._identity import AgentId`
- Never import private modules through full package paths.
  Avoid: `from agentlane.messaging._identity import AgentId`
- Avoid wildcard imports.
- Never place imports inside functions or methods.
- Do not bypass import placement rules with local lint disables.

## Annotation and typing conventions

- Never use `from __future__ import annotations` unless explicitly approved for
  a specific file.
- Never use `TYPE_CHECKING`.
- Never use quoted string annotations. Reorder definitions or extract shared
  types instead.
- Use `TypeAlias` for complex type definitions.
- Use `strenum.LowercaseStrEnum` for string enumerations.
