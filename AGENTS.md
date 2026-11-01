# Agent Guidelines

## Philosophy

This codebase will outlive you. Every shortcut becomes someone else's burden. Every hack compounds into technical debt that slows the whole team down.

You are not just writing code. You are shaping the future of this project. The patterns you establish will be copied. The corners you cut will be cut again.

Fight entropy. Leave the codebase better than you found it.

## Language

Always talk and write documentation in ASD-STE100 Simplified Technical English.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Record design decisions before implementation; scale detail to the change.

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

- Check each requested change against the final diff. Mark a finding resolved
  only when the change is present and verified. Record unresolved items and why.
- Read every changed method, including moved code, against the design and
  readability rules before handing work back.
- Verify the final code after fixes. Check affected public examples when their
  API or behavior changes. Report passed, failed, and skipped checks; distinguish
  mocked verification from live verification. Do not claim completion while
  required checks fail.
- After code changes: spawn the `docs-keeper` agent to detect and update affected READMEs and tech specs
- For public documentation updates, use `.agents/skills/docs-sync/SKILL.md`.
  Apply `dev-docs` to changed prose, then use
  `.agents/skills/docs-portal/SKILL.md` to rebuild and check the website.
  Keep documentation source in root `docs/`.

### 5. Design and simplification

- Before changing a public API or a component's responsibility, put the use
  case, behavior owner, and existing code to reuse in the task plan. Include a
  small client example for API changes. Routine fixes need no separate design.
- Apply the [design and review rules](docs/code-style/design.md) before
  implementation and before handoff. Justify additions with required behavior
  or reachable failures; preserve useful extension points and validation.

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `docs/plans/<task_name>.md` with checkable items
2. **Verify Plan**: Respect requested design review points. Continue already
   authorized work without asking for the same approval again.
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

## Required code readability

- Separate logical steps inside methods with blank lines. Keep statements that
  perform one step together.
- In multi-step methods, put a short comment immediately above each block whose
  purpose, condition, or order needs context. Explain the step's role in the
  operation so the reader can follow the method without tracing every call.
- Put a blank line before a block comment and no blank line between the comment
  and its code. Separate the final return from the preceding work.
- Preserve useful block-purpose comments during simplification. A comment can
  explain application logic; it does not need an external protocol justification.
- Follow the examples and review checklist in
  [`docs/code-style/README.md`](docs/code-style/README.md#logical-blocks-and-comments)
  and [`Python comments`](docs/code-style/python/comments.md#multi-step-methods).

## Local workflow

For code, tests, or build/test changes, follow the
[code-change-verification skill](.agents/skills/code-change-verification/SKILL.md):

```bash
bash .agents/skills/code-change-verification/scripts/run.sh
```

The script runs formatting, lint, type checks, and tests. Documentation-only
changes use the documentation workflow above; they do not require runtime tests.

For a focused test, use `uv run pytest -s -k <test_name>`.

Run Python scripts with `uv run python ...` and Python tools with `uv run`.

## Detailed References

The sections above define the working model. Use the docs under
`docs/code-style/` for implementation details and language-specific rules.

- [`docs/code-style/README.md`](docs/code-style/README.md): index of the
  detailed coding standards.
- [`docs/code-style/workspace-packages.md`](docs/code-style/workspace-packages.md):
  how to add and register a new workspace package.
- [`docs/code-style/python/README.md`](docs/code-style/python/README.md):
  Python conventions, imports and typing, comments, modules, tests, and
  first-party tool design.
- [`docs/code-style/typescript/README.md`](docs/code-style/typescript/README.md):
  TypeScript conventions, strictness, exports, tooling, and package scripts.
