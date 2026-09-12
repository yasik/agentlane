# Native Run Events Over the Process Bridge

Status: B1-B4 complete and verified on 2026-09-12.

## Goal

Make the process bridge transport AgentLane's native serialized run events.
Keep presentation decisions in TypeScript client helpers. Consumers must receive
structured results, full model-event fields, and native event kinds without
maintaining a second Python projection of every run-event type.

## Dependency and Branch

- Depends on the current `yasik/run-event-serialization` work: `RunEvent.to_dict()`
  and `dumps()`, with native kinds and nesting.
- Start a new `yasik/process-bridge-native-events` branch from main after that PR
  merges. Do not add bridge implementation to the event-serialization PR.
- Update Python and TypeScript together. Do not add compatibility modes, protocol
  version churn, or a migration framework for this internal contract change.
  Confirm consumer/release status before implementation; report any new constraint.
- Brain API adoption, persistence, and UI application changes are separate work.

## Baseline Behavior Before B2

`packages/process_bridge/src/agentlane_process_bridge/_events.py` has 12 handlers.
They rename fields, select UI-facing data, derive flags, track turn counts, and
suppress some model events. `RunEvent.to_dict()` instead preserves native structure.

The existing `run_event` is a diagnostic fallback with a flat `run_event_type`
field, as defined in `_events.py` and
`packages/process_bridge_ts/src/protocol.ts`. It does not carry a native record.
The nested `run_event` envelope below was not present in the baseline.

`_protocol.py` also has a permissive `_json_value()` converter: unsupported values,
cycles, and nonfinite numbers become text. Native serialization raises instead.
The writer serves both run events and bridge-owned control events, so replacing
this converter is not a mechanical deletion.

## Proposed Contract

Retain the bridge's version/timestamp envelope and control-event names. Carry each
native event in a single `run_event` envelope, without spreading its fields into
the envelope. Proposed example for review:

```json
{
  "protocol_version": "<existing version>",
  "ts": 0,
  "type": "run_event",
  "event": {
    "type": "model_stream",
    "payload": {
      "kind": "model_stream",
      "event": {"kind": "text_delta", "text": "Hello"}
    }
  }
}
```

The inner example omits other model fields for brevity. The actual `event` must
equal the complete record returned by `to_dict()`. Do not trim, flatten, normalize
reasoning to strings, or drop empty deltas/provider completion records.

- Native errors remain nested model errors. Only the backend's authoritative
  outcome determines whether the whole run failed or completed.
- Approval events retain `payload.event.record`. Host approval commands and the
  broker continue to control execution; UI visibility must not control delivery.
- Stored prompts retain their documented rendered view and values. Actual model
  request inspection uses `RunLLMStartEvent.messages`, not re-rendered final state.
- Preserve lineage. Presentation can distinguish child activity without deleting
  it from the protocol stream.

## Work Units

### B1: Settle the Wire and Failure Contracts

- [x] Confirm the envelope above and how unknown native event kinds reach clients.
- [x] Choose strict native-event serialization: a conversion failure must become
  a controlled bridge run error, cancel/close the active stream, settle pending
  approvals, and never report successful completion for the failed delivery.
- [x] Specify final-result encoding for `run_complete`, which uses `stream.result()`
  and cannot depend on seeing a root `agent_end`. Reuse existing public conversion
  if available; otherwise propose one narrowly scoped shared result API for review.
  Do not import private helpers or wrap arbitrary results in synthetic run events.
- [x] Decide which bridge-owned metadata/control values keep the current text
  fallback. Document any deliberate difference from strict native-event values.
- [x] Review current consumers and full-state payload exposure. Confirm access and
  local-process trust boundaries before forwarding instructions/history/provider data.

Stop for contract review before implementation. No configurable serialization modes.

### B2: Forward Native Events in Python

- [x] Replace `RunEventEncoder` and its per-event handler registry with direct native
  record delivery in `_backend.py`. Remove obsolete public exports and registries.
- [x] Preserve command handling, session reset/configuration, approval resolution,
  cancellation, authoritative completion, queue draining, and broken-pipe cleanup.
- [x] Keep the writer's bounded queue and backpressure. Update batching classification
  to inspect nested model-event kinds, since the top-level type is now `run_event`.
  Lifecycle/control events must retain their flush behavior.
- [x] Apply the approved error/final-result policy from B1. Remove only conversion
  code that has no remaining control-event or metadata callers.
- [x] Do not serialize a full event and then reconstruct the old selected payload.

Primary files: `packages/process_bridge/src/agentlane_process_bridge/_events.py`,
`_backend.py`, `_protocol.py`, and `__init__.py`.

### B3: Decode and Present Native Events in TypeScript

- [x] Add the native-record shape to `src/protocol.ts`, preserving all source fields
  and extras. Validate envelopes and known discriminators without stripping data.
- [x] Update session event dispatch, approval handling, usage reads, and text-stream
  tracking to read nested fields. Remove obsolete flat-event schemas and types.
- [x] Keep convenience helpers for text/reasoning/tool argument deltas only where
  current callers need them. Helpers must not alter protocol event delivery.
- [x] Derive display flags and counts from source fields. Remove bridge-owned turn
  prediction unless an identified client still needs it; otherwise derive locally
  from state snapshots. Do not invent missing lineage/correlation metadata.
- [x] Ensure model completion does not finish the run and model errors do not
  settle unrelated command promises or create duplicate terminal notifications.

Primary files: `packages/process_bridge_ts/src/protocol.ts`, session dispatch,
text-stream tracking, approval handling, and affected public exports.

### B4: Verify and Document

- [x] Generate Python-to-TypeScript fixtures from native records for every run-event
  class, all model kinds, and both approval statuses. Assert exact field retention.
- [x] Test Unicode/newlines, nulls, SDK extras, reasoning signatures, structured tool
  results, full histories, prompt-backed state, and payloads beyond old preview sizes.
- [x] Test the approved unsupported-value/cycle/nonfinite policy for native events,
  final results, and control metadata separately.
- [x] Exercise prompt -> tool -> approval -> resolution -> continuation -> completion
  across the real subprocess boundary, plus cancellation, reset, and shutdown.
- [x] Cover error-after-agent-end, result failure, disconnect, slow pipe, queue
  timeout, and serialization failure without deadlocks or false success.
- [x] Test empty deltas and provider completions remain delivered but need not render.
- [x] Confirm transformed model requests remain distinct from stored prompt views.
- [x] Measure representative full-state payload sizes and serialization/write time
  using synthetic data. Do not add arbitrary truncation to hide increased costs.
- [x] Update bridge protocol docs, TypeScript client docs, examples, and native
  event-serialization docs to describe the unified contract accurately.
- [x] Run `make format`, `make lint`, `make typecheck`, and `make tests`.

## Acceptance

- One native run event produces one intact native record in the transport.
- No Python per-event UI projection remains.
- Typed clients retain complete events independently from rendering.
- Control, approval, cancellation, and stream-result lifecycle tests pass.
- Strict serialization failures have an explicit tested terminal path.
- Native serialization remains independent of transport and snapshot persistence.

## Exclusions

No Brain endpoint changes, persistence implementation, provider routing changes,
new agent orchestration, UI redesign, new event bus, dual-protocol support, release,
or merge authorization. B1 records the approval for this scoped implementation.

## B1 Contract Review — 2026-09-12

The request authorizes implementation and incremental commits. B1 required a
contract review before implementation. The user approved both decisions on
2026-09-12: the public result API and the coordinated protocol 1.0 change.
The following contract governs B2-B4.

### Source and Dependency Checks

- The source plan was found in the main checkout at
  `/Users/yasik/code/personal/agentlane/docs/plans/process-bridge-native-events.md`.
  It was absent from this worktree. This copy retains its work units.
- The clean worktree and fetched `origin/main` both pointed to
  `3df7571ad9edb5cfa1e3606d6469d88ac2d9b686` before the branch was created.
  The native event serializer is present. The implementation branch is
  `yasik/process-bridge-native-events`.
- GitHub reports release `v0.14.0`, published at `2026-09-12T01:22:26Z`.
  The Python and TypeScript manifests both specify `0.14.0`. Package registry
  publication and external installations were not checked.
- Known repository callers are the TypeScript session client and
  `examples/harness/process_bridge_stdio`. The example pins the TypeScript
  package to `0.14.0` while its backend uses this checkout. B4 must make local
  verification use the changed TypeScript package, or it will test mixed versions.
- A source search of the local Brain checkout found no bridge package imports
  or manifest references. This does not establish the absence of external callers.

### Wire and Client Decisions

1. Keep `protocol_version: "1.0"`, `ts`, and bridge control names. Each source
   event becomes exactly one `run_event` with `event = source.to_dict()`.
2. Decode a native record as an object with a string `type` and object `payload`.
   Validate known kinds and their required source fields. Reject contradictory
   known discriminators. Preserve extra fields at every native object level.
   Unknown native kinds remain intact and reach `onEvent`; presentation ignores
   them. Unknown outer bridge event names remain protocol errors.
3. Keep all six current model kinds, including empty deltas, provider records,
   `completed`, and `error`. Only bridge `run_complete`, `run_cancelled`, and
   run-scoped `error` settle the run. Nested model outcomes do not settle commands.
4. Read approvals from `event.payload.event.record`. Deliver request and resolved
   records regardless of display callbacks. Keep the broker as the execution
   authority. Derive presentation flags from source data. Do not add model-event
   lineage where the source has none, and remove predicted turn counts.
5. Batch only nested model kinds `text_delta`, `reasoning`,
   `tool_call_arguments_delta`, and `provider`. Other native records and bridge
   controls drain the queue. Keep the bounded queue, timeout, and write ordering.

### Public Result API Proposal

`RunResult` currently has no public strict conversion method. The strict converter
is private to `harness/_events.py`; transport and trace serializers have different
semantics and cannot replace it.

Add `RunResult.to_dict()` and a public `RunResultRecord` type. The record contains
the complete converted `final_output`, `responses`, `turn_count`, and `run_state`
fields. It has no event kind or transport envelope. Move the existing recursive
conversion into an internal harness module shared by result and event methods.
Do not expose a general object converter or add a second conversion algorithm.

After iteration ends, await `stream.result()` and call its `to_dict()` exactly
once. Build the existing `run_complete` fields from that record: `final_output`,
`turn_count`, response-list length, and the converted final state's `shim_state`.
When `result.run_state` is absent, keep the existing runtime-state shim metadata
fallback. Do not use an `agent_end` event as proof of success.

All fields in the public result record use native strict conversion. Thus an
unsupported response or final state also fails conversion, even if the bridge
completion envelope does not expose that whole field. Document and test this
consequence. The method is an inspection representation, not a snapshot codec.

This adds `src/agentlane/harness/_run.py`, `_events.py`, `__init__.py`, a shared
internal serialization module, and harness serialization tests to B2's files.
Existing event records must remain byte-equivalent after the extraction.

### Failure and Metadata Decisions

- Native records and the complete final result have no text fallback. Unsupported
  values, non-string keys, cycles, nonfinite numbers, and prompt render failures
  fail the run. Convert before enqueueing so a failed record is never partly sent.
  Pass converted native values through the writer's strict payload path.
- On delivery failure, cancel the run token, deny pending approvals, close the
  active stream, and retrieve its result. Settle approvals before waiting for
  cooperative close so an approval waiter cannot prevent teardown.
- With a writable pipe, emit one run-scoped `error` after cleanup and never
  `run_complete`. Preserve the primary failure if close also fails. Test the
  cancellation/reset race during teardown to prevent duplicate terminal events.
- A disconnected or failed writer cannot guarantee terminal delivery. Complete
  local cleanup and let the TypeScript process-exit path settle the active run.
  Do not repeatedly attempt terminal writes or report success after a timeout.
- Keep ready metadata and runtime configuration on their existing strict JSON
  path. Do not make them permissive. Keep the legacy text fallback for ordinary
  bridge-owned payload values, including shim metadata taken from runtime state
  when the result has no state. Native event/result values never use that path.

### Access and Release Constraint

The current transport uses parent/child process pipes. The app chooses the backend
factory and receives stdout. There is no network listener or independent client
authorization layer in this path. Full native events expose instructions, history,
tool results, prompt values, exception messages, and raw provider data to that app.
The host must authorize any later forwarding or storage. The bridge does not
redact these fields. Update the docs without adding an access-control framework.

The existing flat protocol and TypeScript raw-event exports are externally visible.
Keeping protocol version `1.0`, as scoped, makes this a coordinated breaking
package change. Both packages and their callers must update together; mixed
versions are unsupported. No release, publish, compatibility mode, or protocol
version change is authorized by this work.

### Verification and Commit Boundaries

- [x] Inspect the serializer, result type, bridge writer, lifecycle cleanup,
  TypeScript decoder/reducer, local callers, base commit, and release metadata.
- [x] Record concrete wire, result API, failure, metadata, and trust proposals.
- [x] Complete the required B1 contract review with the user.
- [x] B2: Implement Python delivery and the approved public result method;
  update focused Python tests and commit the passing unit.
- [x] B3: Implement TypeScript decoding and presentation; update focused client
  tests and commit the passing unit.
- [x] B4: Add cross-language fixtures, subprocess lifecycle checks, synthetic
  measurements, and docs; run the complete required verification stack and commit.

This review changes documentation only. Verification is source inspection and
`git diff --check`; runtime tests do not prove a proposed contract. The B4
verification requirements remain in force for implementation.

## B2 Implementation Review — 2026-09-12

Native events now pass directly through the strict writer. `RunResult.to_dict()`
shares native conversion and drives authoritative completion. Transport cleanup
wakes blocked input on writer failure and permits process exit with blocked
stdout. Regression tests cover completion/cancel races, approval cleanup, failed
close, and delayed writes after a timeout.

Verification: `uv run pytest packages/process_bridge/tests tests/harness/serialization -q`
passed all 146 tests. Targeted formatting, Ruff, Pyright, and Mypy passed.

## B3 Implementation Review — 2026-09-12

The TypeScript decoder validates native records and retains the original parsed
object, including own reserved-name keys that schema parsing can omit. Client
helpers read native fields and derive presentation data. Nested model completion
and errors do not settle run or command promises. Approval records remain the
execution source of truth.

Verification: `make check-ts` passed lint, type checking, all 84 tests with 284
assertions, and build. This includes eight B4 real subprocess tests, which are
committed with B4. The simplification review removed one unused private field
and corrected one stale comment; it found no required reuse or efficiency change.

## B4 Verification Review — 2026-09-12

The native fixture corpus is generated from Python objects and checked for exact
TypeScript equality. Eight real subprocess tests cover approval, continuation,
completion, failure, cancellation, reset, shutdown, and disconnect. Existing
harness tests retain the distinction between transformed requests and stored
prompt views. The example runs against both packages from this checkout.

The synthetic measurement records 391 bytes for a text delta and about 384 KB
for an agent-end record with 1,000 history entries. The latter measured 0.874 ms
median conversion and 2.951 ms median conversion plus in-memory writer delivery
over 30 repetitions. See `docs/process-bridge/development.md` for all workloads,
commands, and measurement limits. No payload truncation was added.

Required verification passed through
`bash .agents/skills/code-change-verification/scripts/run.sh`:

- `make format` and `make lint` passed. Optional Markdown lint was skipped by the
  Makefile because `markdownlint` is not installed; YAML lint passed.
- `make typecheck` passed Mypy, Pyright, and TypeScript checks.
- `make tests` passed 1,013 Python tests and 84 TypeScript tests (284 assertions).
- `make check-ts` also passed the TypeScript build.
- The local stdio example completed successfully without provider credentials.

The first full check found missing optional Python dependencies.
`uv sync --all-extras --frozen` installed the locked dependencies, and the complete
verification stack then passed. No dependency manifest or root lockfile changed.

### Final Code Review

`ce-code-review` completed with no unresolved findings. Review run:
`20260912-native-bridge-review`. Local artifact:
`/tmp/compound-engineering-501/ce-code-review/20260912-native-bridge-review/review.json`.

Nine local review passes covered correctness, standards, tests, maintainability,
security, performance, API contracts, reliability, and adversarial failures.
These sequential passes are not independent. The external review timed out
without a usable result after a ten-minute task budget; the local adversarial
fallback completed. No independent external confirmation is claimed.

One standards finding was fixed: Python batching now uses the public
`RunEventKind` and `ModelStreamEventKind` enums. The full required verification
stack passed again after that correction, with 1,013 Python and 84 TypeScript
tests. No justified finding or required work remains open.

Commits follow B1, B2, B3, and B4. B2 is `b7cfd6f`; B3 is `daf2258`. B4 contains
the subprocess coverage, measurements, docs, example, and final review fix.
The branch remains local; this work does not publish or release either package.
