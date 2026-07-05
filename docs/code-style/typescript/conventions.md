# TypeScript Conventions

Use these rules for TypeScript across repository packages.

## General Principles

- Keep logic in one function unless it is genuinely composable or reusable.
- Do not extract single-use helpers preemptively. Inline the logic at the call
  site unless the helper is reused, hides a genuinely complex boundary, or has a
  clear independent name that improves the caller.
- Do not reduce line count at the expense of readability, type enforcement, or
  extension clarity. A simplification should remove real duplication or make an
  invariant more explicit; it should not flatten a useful abstraction into a
  looser one.
- Avoid `try`/`catch` where possible. When error handling is required, catch at
  the process, I/O, command, or decode boundary that can report the error
  meaningfully; do not swallow implementation defects in low-level helpers.
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
- Define top-level protocol, command, event, and callback entities as named
  types. Do not pass generic records through application methods and repeatedly
  read fields after ad hoc narrowing; validate raw JSON once at the boundary and
  pass the typed entity after that.

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
- Keep package entrypoints explicit. `src/index.ts` should list the supported
  public surface instead of making internal helpers accidentally importable.

## Variables

- Prefer `const` over `let`.
- Use ternaries or early returns instead of reassignment.

## Control Flow

- Avoid `else`. Prefer early returns.
- Structure a function so its body reads as the happy path.
- Separate distinct guard branches or state transitions with blank lines when
  the spacing makes the decisions easier to scan.

## Helper Placement

- Keep helpers close to the code they support, below the main export when that
  improves readability.
- When extracting helper modules from a complex controller or protocol
  implementation, move the explanatory contract with the code. The new file
  should document what state it owns, what state it intentionally does not own,
  and which lifecycle, settlement, or teardown invariants callers rely on.

## Comments and JSDoc

- Every exported constant, type, class, and function needs concise JSDoc when it
  is part of a package API or protocol surface.
- Public callback types should document when each callback fires, whether raw
  or validated data is passed, and whether the callback is best-effort.
- Internal callback, handler, queue, and adapter types also need JSDoc when
  they define a module boundary. Document each field whose meaning is not
  obvious from its type: who calls it, what state it settles, and whether it can
  close the session, reject a promise, or call app code.
- Public methods on exported classes and important methods on internal
  controller classes should document lifecycle semantics, especially when the
  method accepts backend truth, settles a pending operation, or intentionally
  avoids a local prediction.
- Operation-critical internal constants should have a short comment explaining
  the invariant or lifecycle timing they encode.
- Inline comments should explain protocol boundaries, trust boundaries,
  ordering constraints, failure semantics, and non-obvious branches. Avoid
  comments that restate field names, assignments, or simple control flow.
- Use comments to describe why strict behavior exists. Do not use comments to
  justify implicit data architecture; replace implicit records with named types
  instead.
- Preserve high-quality plan comments during implementation. When a plan
  includes comments that explain ordering, failure, or app-facing contracts,
  treat those comments as part of the implementation and carry them into the
  corresponding code path.
- Avoid orphan helper files that read like fragments. A new TypeScript module
  should make its responsibility clear from the first exported type/class and
  the callback contracts it accepts.

Good comments:

```ts
/** Strict protocol decode failure with actionable payload paths. */
export class BridgeDecodeError extends Error {
  /** Payload field paths involved in the decode failure. */
  readonly fields: readonly string[];
}
```

```ts
// stdout is the protocol channel. Backend diagnostics must use stderr so
// malformed logs never masquerade as bridge events.
const stdout = createInterface({ input: child.stdout });
```

```ts
/**
 * Authoritative runtime config cache for one session.
 *
 * The bridge never predicts config locally. This helper only accepts backend
 * announcements, applies the app decoder, updates the cache, and optionally
 * notifies post-startup subscribers.
 */
export class SessionConfigState<TConfig extends Record<string, unknown>> {
  /**
   * Settle one backend `config` event against the oldest pending configure.
   *
   * Ready/reset config announcements never arrive here; only the `config` event
   * that settles a `configure()` command does. A free-floating `config` event is
   * therefore protocol drift and must close the session loudly.
   */
  settleEvent(event: ConfigEvent, pendingCommands: PendingCommandQueue): void {
    // Failed settlements may still carry a truth snapshot. Apply it before
    // rejecting the promise so the app can immediately re-render backend truth
    // in the catch path.
  }
}
```

```ts
type ReducerHandlers = {
  /** Resolve every cancel() waiter once the run has reached terminal state. */
  settleCancelWaiters: () => void;

  /** Route a backend command error through command FIFO ownership rules. */
  handleCommandError: (message: string) => void;
};
```

Avoid comments like:

```ts
// Get tool name from request.
const toolName = request["tool_name"];
```

Model `request` as a typed entity instead.

## Protocol and Bridge Code

- Treat companion TypeScript packages as strict mirrors of the upstream Python
  protocol when they are released together. Unknown event names, unsupported
  protocol major versions, and missing required fields should fail loudly.
- Do not add unknown-event fallbacks, defaulted required fields, or tolerant
  decoders as future-proofing unless the protocol explicitly promises that
  behavior.
- Use a proven schema library or a small local reader when it makes validation
  stricter and clearer. The goal is one authoritative runtime schema per event,
  not a second loose decoder that can drift from the exported types.
- Keep supported command and event names in one canonical registry. Derive
  known-name lists and type unions from that registry instead of duplicating
  literals.
- Keep extension paths explicit: adding a command or event should require a
  local type/schema entry, processing logic, representative fixture, and parity
  test.
- Avoid hidden sidecar manifests unless they are generated or consumed by real
  tooling.
- Preserve useful class-based or object-based extension seams when they encode
  side-effect ownership, lifecycle state, or implementor obligations. Do not
  flatten them into loosely related functions just to reduce lines.
