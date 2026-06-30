# TypeScript Conventions

Use these rules for TypeScript across repository packages.

## General Principles

- Keep logic in one function unless it is genuinely composable or reusable.
- Do not extract single-use helpers preemptively. Inline the logic at the call
  site unless the helper is reused, hides a genuinely complex boundary, or has a
  clear independent name that improves the caller.
- Avoid `try`/`catch` where possible.
- Never use the `any` type. Reach for `unknown` plus narrowing, or a precise
  type. Biome enforces this mechanically: `noExplicitAny` is a lint error, so
  an explicit `any` fails `make lint-ts`.
- Use Bun APIs when possible, such as `Bun.file()`.
- Rely on type inference inside function bodies, but annotate return types on
  exported functions. Biome's `useExplicitType` rejects missing explicit types
  as a lint error.
- Prefer functional array methods such as `flatMap`, `filter`, and `map` over
  `for` loops.
- Reduce total variable count by inlining a value that is used only once.

## Destructuring

- Avoid unnecessary destructuring. Use dot notation to preserve context.

## Imports and Exports

- Never use value star imports. Do not write `import * as Foo from "..."`.
  Biome enforces this mechanically: `noNamespaceImport` is a lint error, so a
  value star import fails `make lint-ts`.
- Never alias imports (`import { foo as bar } from "..."`) and never use a
  type-only star import (`import type * as Foo from "..."`). Biome has no rule
  for either form, so these remain review-enforced discipline.
- Never re-export with `export *`; list the names you re-export. Biome's
  `noReExportAll` rejects wildcard re-exports.
- Prefer dynamic imports for heavy modules that are only needed in selected code
  paths, especially on startup-sensitive entry points.

## Variables

- Prefer `const` over `let`.
- Use ternaries or early returns instead of reassignment.

## Control Flow

- Avoid `else`. Prefer early returns.
- Structure a function so its body reads as the happy path.

## Helper Placement

- Keep helpers close to the code they support, below the main export when that
  improves readability.
