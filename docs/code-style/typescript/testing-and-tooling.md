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
- For protocol and bridge packages, add parity tests that compare the canonical
  command/event registry with fixtures, decoder schemas, process wiring, and
  any Python-side vocabulary exported for downstream packages.
- Add representative fixtures for every supported protocol event. Decode them
  strictly in tests so drift in required fields fails before an app consumes the
  package.
- Test malformed payloads as hard failures, including unknown event names,
  missing required fields, nested payload drift, and unsupported protocol major
  versions.

## Type Checking

- Type-check with `make typecheck-ts` rather than invoking the compiler ad hoc.
- `tsconfig.json` runs in `strict` mode with `moduleResolution: "Bundler"` and
  Bun types. Keep it strict; do not silence errors with `any`, `@ts-ignore`, or
  `@ts-expect-error` unless you document why in a comment.
- Published TypeScript packages must also keep a package-local build script that
  emits `dist/` JavaScript and declaration files. Treat `bun run build` as part
  of the public package contract, not only a release-time convenience.

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

- Run `make check-ts` for TypeScript-only changes; it runs lint, type check,
  tests, and the package build together.
- Run the full repository verification script before marking runtime, protocol,
  or package-tooling changes done.
- If a TypeScript package adds a dependency, update its package manifest and
  lockfile in the same change and keep the dependency justified by a real
  simplification, validation, or runtime need.
- Keep commit messages concise and in the imperative mood.
- Do not mention "Co-Authored" or "Authored By" in commit or PR text.
