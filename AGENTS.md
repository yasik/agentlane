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

## Detailed References

The sections above define the working model. Use the docs under
`docs/code-style/` for implementation details and language-specific rules.

- [`docs/code-style/README.md`](docs/code-style/README.md): index of the
  detailed coding standards.
- [`docs/code-style/imports.md`](docs/code-style/imports.md): import paths,
  annotation rules, and typing boundaries.
- [`docs/code-style/workspace-packages.md`](docs/code-style/workspace-packages.md):
  how to add and register a new workspace package.
- [`docs/code-style/python.md`](docs/code-style/python.md): Python conventions,
  async patterns, error handling, Pydantic/FastAPI guidance, and tooling rules.
- [`docs/code-style/comments.md`](docs/code-style/comments.md): comment,
  docstring, and field comment expectations.
- [`docs/code-style/modules.md`](docs/code-style/modules.md): file naming,
  package layout, `__init__.py`, and export rules.
- [`docs/code-style/testing-and-prs.md`](docs/code-style/testing-and-prs.md):
  test naming, fixtures, PR expectations, and commit message guidance.
