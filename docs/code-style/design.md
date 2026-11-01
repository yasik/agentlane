# Design and review

Use these rules for changes to framework behavior, public APIs, and component
responsibilities. Apply checks to the affected path; do not expand a routine
fix into unrelated refactoring.

## Ownership

- AgentLane owns reusable runtime and harness contracts. Applications own
  business decisions, tenant rules, user interfaces, and deployment policy.
  Storage interfaces and reusable defaults can belong in the framework;
  application retention and ingestion workflows belong in the application.
- Extend the runtime, model, tool, state, or shim primitive that already owns
  the behavior before creating a parallel implementation. Keep provider
  adapters thin by using the SDK and shared harness behavior.
- Shared primitives belong in the core. The core must not require an optional
  extension to provide its foundation.
- Document framework defaults and expose typed extension points where callers
  need different behavior. Do not make one application's assumptions generic.
- Give each mutable state value one authoritative owner. Pass required context
  explicitly instead of introducing ambient state to avoid a contract.

## Simplicity and client setup

- Check the smallest realistic caller, including configuration and result
  handling. Supply useful defaults for common supported workflows so callers
  do not have to repeat framework parsing, orchestration, or validation.
- Inspect existing primitives and dependencies before writing custom machinery.
  Use a proven library when its benefit justifies its cost and coupling.
- Add helpers, classes, and types for distinct responsibilities, state ownership,
  reuse, or meaningful extension contracts. Avoid pass-through helpers and
  parallel types that only rename the same data. Preserve readability, type
  enforcement, and useful extension points when reducing code.
- Validate external input at the owning boundary and pass concrete types inside
  it. Remove checks for states made impossible by construction. Keep validation
  where mutable state, capabilities, or external output can still change.
- Copy state only to protect an identified mutation, snapshot, or rollback
  boundary. Another layer receiving the data does not itself require a copy.
- Add compatibility paths only for an identified supported consumer or version.
  Preserve established contracts without inventing aliases, migrations, or
  fallbacks for hypothetical consumers.

## Complete behavior

- For prompt, tool, history, compaction, or adapter changes, trace a representative
  request through the normal application path. Inspect the final model request,
  committed state, and client events as separate outputs.
- Check message order and roles, tool-call/result pairing, retained metadata,
  and duplicate context. Make intentional transformations explicit. Preserve
  stable request prefixes on ordinary turns and check restore and continuation
  when persisted state changes.
- Apply output limits where the contract owns them. A documented model-facing
  tool cap can be valid; transport must not silently truncate authoritative data
  to satisfy a display preference.
- Distinguish operational failures, invalid configuration or protocol data,
  programming defects, and cancellation. Handle them at the boundary that owns
  recovery. The [tool error contract](./python/tool-design.md#error-results)
  applies to model-facing handlers, not every framework function.
- Preserve cancellation and cleanup. Do not turn cancellation into success or
  a normal tool error, or commit partial state as a completed run.

## Useful evidence

- Reuse tests that already prove the behavior. Add tests for changed contracts,
  demonstrated regressions, or reachable failures with material consequences.
  Make the purpose clear in the test name or a short comment.
- Do not add tests that repeat implementation logic, assert incidental helper
  structure, or only increase coverage. Do not retain unnecessary production
  code to satisfy tests for invented states.
- Test contracts at public boundaries. Use focused internal tests for substantial
  algorithms or state transitions, without a large mocked application setup.
- For affected async paths, verify cancellation, early close, cleanup, and state
  commitment at the owning boundary. Existing coverage can satisfy this check.
- Verify new common public workflows with small runnable examples. Separate
  presentation code from essential framework setup and state which dependencies
  are mocked. Use the repository verification workflow for final checks.
