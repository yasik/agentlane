# Compaction

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
    CompactionShim,
    CompactionShimConfig,
    Compactor,
    ContextSignal,
    DefaultCompactor,
    DefaultCompactorConfig,
    estimate_message_tokens,
    is_summary_item,
    render_request_messages,
    render_summary_item,
)
```

The package exports the stock `CompactionShim`, configuration types,
request/result dataclasses, observer report types, summary helpers, a default
estimator, the `Compactor` protocol, and the stock `DefaultCompactor`.

## Configuration

`CompactionShimConfig` defines trigger and failure behavior for a compaction
shim:

1. `context_window` is the protected model context window.
2. `trigger_ratio` chooses the threshold as a fraction of `context_window`.
3. `trigger_tokens` sets an absolute threshold and takes precedence over
   `trigger_ratio`.
4. `on_failure` is either `"inject"` or `"skip"`.
5. `name` is the stable prefix for compaction reports and attempt keys.
6. `default_max_tokens` sets the main-model output cap when `model_args` has
   no explicit output limit. It defaults to `None`. A positive integer adds
   `max_tokens` to the prepared model arguments.

Use `resolved_trigger_tokens()` when the shim needs the absolute threshold.

`DefaultCompactorConfig` carries the shared settings for summary-plus-tail
compactors: the summarization prompt, summary bridge, recent-history budget,
summary placement, and optional summary output cap.

## Default Compactor

`DefaultCompactor` is the stock summary-plus-tail implementation. It keeps a
recent history tail using both `keep_recent_messages` as a minimum history-item
floor and `keep_recent_tokens` as an approximate token budget, summarizes older
history with the model supplied on `CompactionRequest`, and returns replacement
history with one tagged summary item plus the retained tail.

The compactor always retains the newest block that is neither internal context
nor a summary, plus any internal messages after that block. This rule applies
even when those items exceed the tail budget. Internal messages remain separate
blocks; older internal messages follow the normal message and token budgets.
If history contains only internal context, the compactor protects its newest
block. Retained internal messages keep their runtime metadata; the summary
model receives their contents without runtime metadata fields.

The compactor always keeps
[ephemeral messages](../models/overview.md#ephemeral-context), because the
active run still needs them:

- The summary model never receives an ephemeral message, and
  `CompactionResult.dropped_items` never contains one.
- An ephemeral message in the retained tail keeps its position. An older
  ephemeral message moves to the start of the retained tail, in its original
  order. This also applies to an ephemeral message inside a tool-call block.
- Ephemeral messages do not anchor the tail and do not count toward
  `keep_recent_messages` or the tail token count.
- The compactor subtracts the estimated tokens of ephemeral messages from the
  tail token budget, in the same way as the instructions and the output cap.

```python
from agentlane.harness.compaction import DefaultCompactor, DefaultCompactorConfig


compactor = DefaultCompactor(
    DefaultCompactorConfig(
        keep_recent_tokens=20_000,
        keep_recent_messages=4,
        summary_max_tokens=4_096,
    )
)
result = await compactor.compact(request)
```

The default compactor does not decide when compaction should run. A shim still
owns trigger evaluation, failure policy, observer reports, and the final
`turn.replace_history(result.history)` call.

The summary output cap is independent of the main-model output cap. The
compactor removes `max_tokens`, `max_output_tokens`, and `max_completion_tokens`
from the copied request arguments, then sets `max_tokens` to
`summary_max_tokens`. LiteLLM clears constructor-level output-limit aliases
when a call supplies an output cap. With `summary_max_tokens=None`, the
compactor sends no output override, so constructor defaults can still apply.
Other `model_args` pass through to the summarizer call. Use a custom compactor
to change those arguments. Each compaction replaces prior summary items with
one summary.

For preemptive or manual requests below the trigger threshold, the compactor
returns a no-op result when there are fewer than the minimum older blocks to
summarize. Triggered automatic requests are allowed to summarize fewer blocks
because the request is already at the configured threshold. It raises
`ContextOverflowError` when the summarizer request cannot fit the configured
context window. It raises `CompactionError` when the model returns an empty
summary, the finish reason is not `stop`, or the replacement does not shrink
the estimated request meaningfully. Incomplete summaries are never installed.

## Compaction Shim

`CompactionShim` is the stock runtime shim. It evaluates the configured trigger
before a model turn, calls a compactor when the request is at or above the
threshold, installs successful replacement history with
`turn.replace_history(...)`, and emits optional `CompactionReport` values.

```python
from agentlane.harness.compaction import (
    CompactionShim,
    CompactionShimConfig,
    DefaultCompactor,
)


shim = CompactionShim(
    CompactionShimConfig(context_window=128_000),
    DefaultCompactor(),
)
```

By default, the shim uses `DefaultCompactor`, `estimate_message_tokens`, and
`on_failure="inject"`. A compactor error leaves the original history unchanged
and adds an internal user-role note to the next model request. The run continues
if the final request fits the context window. Set `on_failure="skip"` to omit
the note. Pass `on_compact=...` to
receive one `CompactionReport` for each triggered attempt, including no-op
attempts and failures. Compactor failures are reported immediately. Successful
compactor results are reported after all transforms and validators finish.
These reports include the recovered history count and the final token estimate
for messages, tools, and output. If final request validation fails, the report
has `compacted=False` and an error. Observer callback errors are ignored and do
not change the compaction result.

At the start of each invocation, the shim captures the most recent user
message that is neither a summary nor internal context. It stores that exact
message in the agent's persisted shim state. If compaction removes the message,
the shim restores it before the retained assistant and tool history. Each agent
keeps its own captured request, including across JSON snapshot restores.

`SkillsShim` and `HarnessToolsShim` restore missing active skill instructions
and the current plan from their own state. They append recovery context only
when it is absent. Ordinary turns retain the previous request prefix.

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

The stock shim includes system instructions, history, tool schemas, and an
output allowance in its trigger estimate. During preparation, the shim removes
null output limits and validates `max_tokens`, `max_output_tokens`, and
`max_completion_tokens` as positive integers. Other invalid limits raise an
error. If no explicit limit remains, it adds `max_tokens=default_max_tokens`
when that default is set. The output allowance is the maximum of the prepared
output limits. Configure `default_max_tokens` or an explicit limit to include
the main-model output in the budget.

After all message transforms, the shim checks the final messages, tool schemas,
and output allowance against `context_window`. This includes restored plan and
skill context and context injected by other shims. If the request is too large,
the harness restores the history from before turn preparation and raises
`ContextOverflowError` before the main-model call. System instructions remain
complete. Use a smaller configured window to leave room for estimation error
and provider framing.

Exact accounting depends on the provider and model. A compaction shim can pass a
custom `TokenEstimator` into `CompactionRequest` when it has access to a
provider-specific tokenizer or usage API. The stock shim records the latest
provider-reported usage when available, but it does not let that reported value
trigger compaction by itself when the locally rendered request is still below
threshold. That avoids false compaction after handoffs or resumes where older
responses may have been produced by a different model context.

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

A custom compactor must keep each ephemeral message in
`CompactionResult.history` and must not send it to a summarizer. Use
`is_ephemeral_message(item)` from `agentlane.models` to find these messages.
The harness does not check this rule. If a compactor removes an ephemeral
message, the rest of the run does not receive it.

## Shim Integration

Persistent compaction is a history rewrite. The shim should install a
successful `CompactionResult.history` with:

```python
turn.replace_history(result.history)
```

`replace_history(...)` copies the provided items before installing them, so a
caller can safely mutate local result objects after the prepared turn is
updated.

Place `CompactionShim` last in the descriptor's shim list. Its trigger then
includes history, tools, and restored context supplied during earlier
`prepare_turn(...)` callbacks. Final request validation runs after every
`transform_messages(...)` callback, regardless of descriptor order.

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
