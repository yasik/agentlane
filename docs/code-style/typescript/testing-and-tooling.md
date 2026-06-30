# Testing and Tooling

Use this page for TypeScript test conventions, type checking, and the commands
that gate a change.

## Testing

- Run package TypeScript tests through the repository target:
  `make test-ts`. Tests live in package-local `tests/` directories as
  `*.test.ts` / `*.test.tsx`.
- Use package-local `bun run test` only for a focused inner loop. The final
  verification path should use the repository `make` target or the full
  verification script.
- Avoid mocks as much as possible.
- Test the actual implementation. Do not duplicate the logic under test into the
  test.

## Type Checking

- Type-check with `make typecheck-ts` rather than invoking the compiler ad hoc.
- `tsconfig.json` runs in `strict` mode with `moduleResolution: "Bundler"` and
  Bun types. Keep it strict; do not silence errors with `any`, `@ts-ignore`, or
  `@ts-expect-error` unless you document why in a comment.

## Formatting

Biome owns formatting. Run `make format-ts`. Match these defaults as you write
so the formatter is a no-op:

- 2-space indentation, 80-column line width, LF line endings.
- Double quotes, semicolons always, trailing commas everywhere.
- Always parenthesize arrow-function parameters.

## Linting

Run `make lint-ts`. Warnings fail the gate, so warn-level rules and future
nursery warnings block a change just like errors. Rules Biome enforces as
errors:

- Use `===` / `!==`, never `==` / `!=` (`noDoubleEquals`).
- No `debugger` statements (`noDebugger`).
- No `eval` (`noGlobalEval`).
- No `export *` re-exports (`noReExportAll`).
- No explicit `any` (`noExplicitAny`).
- No unused imports or variables (`noUnusedImports`, `noUnusedVariables`).
- Explicit types on exported boundaries, including return types
  (`useExplicitType`).
- Keep files under 500 lines (`noExcessiveLinesPerFile`). Split a module before
  it grows past that.

Package-local Biome configs should allowlist the package `src/` and `tests/`
trees they own.

## Before You Open a PR

- Run `make check-ts` for TypeScript-only changes; it runs lint, type check, and
  tests together.
- Run the full repository verification script before marking runtime changes
  done.
- Keep commit messages concise and in the imperative mood.
- Do not mention "Co-Authored" or "Authored By" in commit or PR text.
