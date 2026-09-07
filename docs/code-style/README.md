# Code Style Index

These pages hold the detailed coding conventions referenced by `AGENTS.md`.
Read the page that matches the work you are doing instead of treating
`AGENTS.md` as a long-form style manual.

## Pages

1. [Python conventions](./python/README.md): Python style, imports and typing,
   comments, modules, tests, and first-party tool design.
2. [Workspace packages](./workspace-packages.md): how to add and register a new
   workspace package.
3. [TypeScript conventions](./typescript/README.md): TypeScript tooling,
   strictness, exports, and package script expectations.

## Logical blocks and comments

These rules apply to all code and tests in the repository.

- Use a blank line between blocks that do different work, such as input
  checks, conversion, metadata handling, side effects, and result construction.
- Separate an early-return or `continue` guard from the next logical block.
  Separate a completed loop from the final return when they are distinct steps.
- Keep related statements together. Do not put a blank line after every line,
  and do not compress separate decisions to reduce the line count.
- Put a short comment above a block when its purpose, ordering constraint,
  or external protocol rule is not clear from the code. Use these comments
  inside functions as well as at module and function boundaries.
- Explain why the block exists or which constraint it preserves. Do not
  narrate assignments or add comments to trivial branches.
- Review logical spacing and comment accuracy after formatting. A formatter
  cannot decide where one idea ends and the next begins.
