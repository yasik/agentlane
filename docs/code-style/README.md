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
4. [Maintain documentation](./documentation.md): root Markdown, writing review,
   and documentation portal checks.

## Logical blocks and comments

These rules are required for all code and tests in the repository.

- Use a blank line between blocks that do different work, such as input
  checks, conversion, metadata handling, side effects, and result construction.
- Separate an early-return or `continue` guard from the next logical block.
  Separate a completed loop from the final return when they are distinct steps.
- Keep related statements together. Do not put a blank line after every line,
  and do not compress separate decisions to reduce the line count.
- In multi-step methods, use short comments to explain the purpose of the
  preparation, conditional processing, restrictions, and event handling that
  form the operation. A method name alone does not always explain why that step
  belongs here or why it runs in this order.
- Put each comment immediately above its block. Leave a blank line before the
  comment, but keep the comment attached to its code. Separate the final return
  from the preceding work.
- Explain the block's role, the reason for a condition, or the constraint it
  preserves. Comments can explain application logic as well as external
  protocol rules. Do not narrate individual assignments or label every line.
- Keep useful block-purpose comments during refactoring and simplification.
  Update them when behavior changes. Do not remove them only because the code
  is internal or the comment does not describe an external constraint.
- Review logical spacing and comment accuracy after formatting. A formatter
  cannot decide where one idea ends and the next begins.

Before completing a code change, read each changed method from top to bottom.
Check that each logical step is visually separate and that its purpose and
order are clear. Add context at the block that needs it. Do not extract a
single-use helper only to avoid explaining the step where it occurs.

See the [Python example](./python/comments.md#multi-step-methods) for comment
placement and spacing.
