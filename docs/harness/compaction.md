# Harness Compaction

Conversation compaction is the harness-level pattern for replacing older
conversation history with a concise summary while keeping recent history
available to the next model call.

AgentLane exposes the public contracts and helper functions for this pattern in
`agentlane.harness.compaction`. A compaction implementation usually lives in a
shim: the shim decides when to compact, asks a `Compactor` for replacement
history, then installs that replacement through `PreparedTurn.replace_history`.

## Import Path

```python
from agentlane.harness.compaction import (
    CompactionRequest,
    CompactionResult,
    CompactionShimConfig,
    Compactor,
    ContextSignal,
    DefaultCompactorConfig,
    estimate_message_tokens,
    is_summary_item,
    render_request_messages,
    render_summary_item,
)
```

The package exports configuration types, request/result dataclasses, observer
report types, summary helpers, a default estimator, and the `Compactor`
protocol.

## Configuration

`CompactionShimConfig` defines trigger and failure behavior for a compaction
shim:

1. `context_window` is the protected model context window.
2. `trigger_ratio` chooses the threshold as a fraction of `context_window`.
3. `trigger_tokens` sets an absolute threshold and takes precedence over
   `trigger_ratio`.
4. `on_failure` is either `"raise"` or `"skip"`.
5. `name` is the stable shim-state key prefix for one compaction instance.

Use `resolved_trigger_tokens()` when the shim needs the absolute threshold.

`DefaultCompactorConfig` carries the shared settings for summary-plus-tail
compactors: the summarization prompt, summary bridge, recent-history budget,
summary placement, and optional summary output cap.

## Request Rendering

Compaction decisions should use the same canonical request shape that the
runner sends to the model. Use:

```python
messages = render_request_messages(turn.run_state.instructions, turn.run_state.history)
```

The helper accepts the same run-history item types as the runner, including
canonical message dicts, `ModelResponse` values, `PromptSpec` values, strings,
JSON-like values, and Pydantic models.

This renderer is the public compaction facade over the runner's canonical
message builder. It keeps token estimates and summarizer input aligned with the
actual next request.

## Token Accounting

`estimate_message_tokens(messages)` is a local preflight estimator. It uses a
UTF-8 byte heuristic for text and a fixed charge for non-text content parts.

Exact accounting depends on the provider and model. A compaction shim can pass a
custom `TokenEstimator` into `CompactionRequest` when it has access to a
provider-specific tokenizer or usage API.

`ContextSignal` records one trigger evaluation. It includes the effective token
estimate, optional server-reported token count, instruction-only estimate,
configured window, trigger threshold, source of the signal, current turn count,
and history item count.

## Summary Items

Compaction summaries are stored as ordinary user-role message dicts marked with
stable summary delimiters.

Use `render_summary_item(bridge=..., summary_text=...)` to create the history
item, and `is_summary_item(item)` to detect an existing summary during later
compaction passes.

The default summary item template is Jinja2-rendered and exported as
`DEFAULT_SUMMARY_ITEM_TEMPLATE`. The summary markers are also exported for
advanced integrations that need to inspect the raw content.

## Custom Compactors

A custom compactor implements the `Compactor` protocol:

```python
from agentlane.harness.compaction import (
    CompactionRequest,
    CompactionResult,
    render_summary_item,
)


class TailOnlyCompactor:
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        tail = list(request.history[-4:])
        dropped = request.history[:-4]
        summary = render_summary_item(
            bridge="The earlier conversation has been summarized.",
            summary_text="Earlier turns were omitted by this compactor.",
        )
        return CompactionResult(
            history=[summary, *tail],
            summary_content=str(summary["content"]),
            dropped_items=dropped,
            summarizer_response=None,
        )
```

Production compactors normally call `render_request_messages(...)` before
summarizing so the summarizer sees the same canonical conversation the next
runner turn would have seen.

## Shim Integration

Persistent compaction is a history rewrite. The shim should install a
successful `CompactionResult.history` with:

```python
turn.replace_history(result.history)
```

`replace_history(...)` copies the provided items before installing them, so a
caller can safely mutate local result objects after the prepared turn is
updated.

Descriptor order still matters. Place a compaction shim after shims that append
history for the current turn when the compactor should account for those
additions.

`transform_messages(...)` remains the one-call escape hatch for changing the
final request. Use `replace_history(...)` for resumable compaction because it
updates the persisted `RunState.history`.

## Boundaries

Compaction rewrites `RunState.history`. It does not rewrite
`RunState.responses`; raw model responses remain the record of completed model
calls.

Run events also use the word "compact" for `RunStateSnapshotEvent`, which means
the event payload is small. That event snapshot is separate from conversation
compaction.
