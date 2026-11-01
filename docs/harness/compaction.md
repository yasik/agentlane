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
recent history tail using `keep_recent_messages` and `keep_recent_tokens` as
recency targets within the hard preservation budget, summarizes older
history with the model supplied on `CompactionRequest`, and returns replacement
history with one tagged summary item plus the retained tail.

The compactor always retains the newest block that is neither internal context
nor a summary, plus any internal messages after that block. This rule applies
even when those items exceed the tail budget. Standalone prior compaction
context envelopes are excluded. Internal messages outside tool rounds remain
separate blocks; older internal messages follow the normal message and token
budgets.
If history contains only internal context, the compactor protects its newest
block. Retained internal messages keep their runtime metadata; the summary
model receives their contents without runtime metadata fields.

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
before an explicit model request, calls a compactor when the request is at or
above the threshold, installs successful replacement history with
`turn.replace_history(...)`, and emits optional `CompactionReport` values.

An automatic successor that the provider has already started keeps its accepted
input, including transient context from message transforms. Compaction and
final request validation resume before the next explicit request. Provider
context and output limits still apply to automatic successors.

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
and adds a non-persisted internal user-role note to the next model request.
The run continues if the final request fits the context window. Set
`on_failure="skip"` to omit the note. Pass `on_compact=...` to receive one
`CompactionReport` for each triggered attempt, including no-op
attempts and failures. Compactor failures are reported immediately. Successful
compactor results are reported after all transforms and validators finish.
These reports include the recovered history count and the final token estimate
for messages, tools, and output. If final request validation fails, the report
has `compacted=False` and an error. Ordinary observer callback errors are ignored
and do not change the compaction result.

At the start of each invocation, the shim captures the most recent user
message that is neither a summary nor internal context. It stores that exact
message in the agent's persisted shim state. During the invocation, delivered
native steering input updates this saved request through `on_input_messages`.
Pending steering receipts and application history edits do not trigger this
callback. The shim does not scan history again before each turn. Each agent
keeps its own captured request, including across JSON snapshot restores.

The shim supplies the exact saved request through
`CompactionRequest.required_messages` before compaction. `DefaultCompactor`
retains its original block, or includes a missing anchor in the replacement.
It selects all retained history and current state before the summary call,
assembles one candidate, then checks shrinkage. The shim never repairs a
compactor result by adding messages afterward. A changed custom result that
omits a required message is rejected. A no-op result leaves history unchanged.
Final validation runs after successful, failed, and no-op compaction attempts.

`HarnessToolsShim` supplies the current plan as owner context for selection.
Skill instructions are excluded from automatic preservation and reload on
demand through `activate_skill`. Ordinary turns retain the previous request
prefix.

## Preserve application state

`PreservedContext` keeps application state available to the model when
compaction removes its conversation messages. State can remain in storage while
the generated summary omits details needed to continue the task.

The task plan is one example. `HarnessToolsShim` stores the authoritative plan
in `RunState.shim_state`. If compaction removes the `write_plan` call, that
stored plan survives, but the next model request can lose its exact steps and
statuses. During `prepare_turn`, the tool shim adds a `PreservedContext` to
`turn.preserved_context` with the complete rendered plan and references to any
messages that still establish that plan.

Compaction uses that projection in its preservation decision. If a message
listed in `represented_by` is retained, compaction avoids injecting a duplicate.
The match uses exact message equality. Otherwise, it injects the selected plan
text as internal context and counts it against the token budget before
summarization. The default and
ranked selectors keep these owner projections; a custom selector can omit them.

The state owner remains responsible for storage and updates. `PreservedContext`
carries its text for one preservation decision. This separates ownership from
retention: the tool shim supplies the current plan, and compaction selects and
budgets its representation. Shims can supply other application state through the
same contract without adding state-specific logic to the compactor.

## Select preserved context

Pass a `PreservationSelector` to `DefaultCompactor` to replace or extend its
default recency selection. This protocol defines
`async __call__(request) -> PreservationSelection`; async functions and callable
objects satisfy it without inheritance.

The selector receives a `PreservationRequest` with detached
rendered instructions, history blocks, exact required messages, and the default
selection. `request.estimate_tokens(selection)` uses the same estimator and
canonical material as the compactor's preflight check. The request data is
detached; the estimate callback uses the original blocks and shared estimator.
Keep the estimator free of side effects. The callback validates the selection
and raises `CompactionError` for invalid IDs or context instead of returning a
token count.

Each `PreservationBlock` has an attempt-local `id`, complete `messages`, an
estimated `tokens` cost, and two constraints:

- Include every block with `required=True`.
- Select only blocks with `retainable=True`. Standalone prior summaries and
  compaction context envelopes cannot be retained as original blocks.

Return a `PreservationSelection(block_ids=..., context=...)`. IDs must be unique
and valid for this attempt. Do not persist them or reuse them on a later
attempt. Selected blocks retain their original chronology and complete tool
interactions, regardless of ID order in the selection. Edits to callback message
copies do not change retained history. Selection preserves existing tool groups;
it does not repair malformed original history.

An atomic tool block includes the span from a call through its matching
results, including intervening steering messages and context. Overlapping tool
rounds stay together, so a selector cannot retain a result without its call.
Required blocks cannot be removed or split.

Start with `request.default.block_ids` and `request.default.context` to extend
the defaults. To replace them, choose your own retainable blocks and context,
while keeping all required blocks. `context` is a required constructor argument:
pass `request.default.context` to keep owner projections or explicitly pass
`()` to omit them. Each `PreservedContext` needs a unique `origin` and complete
`content`. Optional `represented_by` messages must each contain the full state:
one retained match suppresses that state entry. Each representation must still
establish current state after later updates; an earlier equal value is insufficient.
Selected entries are assembled
into one internal message with origin `agentlane.compaction.context`. Each entry
retains its owner label and complete content. Standalone prior context envelopes
are summarized and replaced, including when the next selection uses `context=()`.
The `PreservedContext.origin` field labels the owner within this envelope. It
does not identify history messages to replace.

`token_budget` is the maximum cost of retained blocks and injected context
after reserving system instructions, main-model tools and output, summary
output, and the summary envelope. Use `request.estimate_tokens(selection)` to
count required messages absent from selected blocks and context that needs
injection without double counting represented state. The compactor checks the
full selection before calling the summary model. The default selection trims
optional tail blocks to fit the hard budget. Required anchors and complete
tool groups can still make a selection impossible; that attempt fails without
a summary call or history replacement.

`retention_budget` is the soft allowance from `keep_recent_tokens`, reduced when
needed to leave room below the compaction trigger. Required anchors can exceed
this allowance, but the complete selection must still fit `token_budget`.

A pending tool call and its intervening messages can form an indivisible block
that exceeds the budget. Automatic successors do not split or compact that
block. If compaction cannot produce a valid replacement and the next explicit
request exceeds `context_window`, the harness raises `ContextOverflowError`,
preserves the history from before turn preparation, and cancels outstanding
tool tasks.

Older evidence selected for exact retention also remains available to the
summarizer. Only the contiguous retained recent suffix is excluded from its
input. This keeps an older selected result connected to the surrounding work.

To select blocks with a decision model, pass a configured `Model[ModelResponse]`
as `decision_model` to `ModelPreservationRanker`:

```python
from agentlane.harness.compaction import (
    DefaultCompactor,
    ModelPreservationRanker,
    RankedPreservationSelector,
)

compactor = DefaultCompactor(
    selector=RankedPreservationSelector(ModelPreservationRanker(decision_model)),
)
```

The built-in ranker prepares the input, supplies the response schema, calls the
model, and parses the returned block IDs. It rejects incomplete responses and
invalid JSON output before summarization. It skips the model call when no
optional retainable blocks are available. The compactor still uses the agent's
configured model to generate the summary; `decision_model` only ranks blocks.

Optional: pass `prompt` to replace the relevance policy in
`DEFAULT_RANKING_PROMPT`. The ranker always supplies the JSON output contract
and instructions to treat conversation content as data. You only specify what
evidence to prioritize:

```python
ranker = ModelPreservationRanker(
    decision_model,
    prompt="Prioritize unresolved errors and their evidence.",
)
```

The serialized `RankingPayload` contains `instructions`, `required_messages`,
`current_state`, and `blocks`. It includes every block in `request.blocks`.
Each `RankingPayloadBlock` contains `id`, `required`, `retainable`, and complete
`messages`. Blocks with `retainable=False` provide background for ranking,
including standalone prior summaries and state projections. The model must
return only optional block IDs with `retainable=True`. Current owner state takes
precedence over earlier projections.

`required_messages` adds only exact anchors absent from these blocks. The stock
shim supplies the latest genuine user message as an anchor; custom callers can
supply more.

`current_state` adds an owner entry only when none of its `represented_by`
messages occur in any block or added anchor, including background blocks. These
checks use exact message equality. This input deduplication leaves
`request.default.context` unchanged.
The selector still injects state when its representing block is not retained.

Configure timeouts, retries, and output limits on `decision_model`. The ranker
sends the full payload in one request without batching or trimming. The ranking
model must support that input size; the preservation token budget applies to
retained context, not to the ranking request.

Pass the ranker to `RankedPreservationSelector`. This concrete selector keeps all
required blocks and `request.default.context`, then adds ranked optional blocks
that fit both `retention_budget` and `token_budget`. It replaces the default
recency preference. An omitted optional block is not retained. A block that
exceeds the budget is skipped so a
later block can still fit. Already-selected IDs are ignored. Invalid types,
unknown IDs, and unretainable IDs raise `CompactionError` before summarization.

The framework owns budget checks and state deduplication. Retaining a block can
reduce the cost of injected state, so the compactor validates the completed
selection before summary generation. To change current-state selection or extend
the default recent tail, implement `PreservationSelector` directly.

For a custom decision API or application rules, implement the
`PreservationRanker` protocol with `async rank(request) -> Sequence[int]` and
pass that object to `RankedPreservationSelector`. Return retainable block IDs
from highest to lowest relevance. Both protocols receive `PreservationRequest`
as a positional argument and require no inheritance. A custom ranker owns its
API calls, response parsing, and fallback policy. Treat conversation text as data
and apply the application's data-access rules. Uncaught ranking errors use the
compaction failure policy.

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
and output allowance against `context_window`. This includes selected plan
context and messages injected by other shims. If the
request is too large, the harness restores the history from before turn
preparation and raises `ContextOverflowError` before the main-model call.
System instructions remain
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

The system skills catalog remains the source for skill discovery after
compaction. The default summary prompt tells the model to omit skill bodies
and the catalog. The shim does not add a cumulative skill-use list to summaries
or restore skill bodies automatically. When full instructions are missing and
needed, the model can request them through `activate_skill`.

Compaction summaries are internal user-role messages with stable summary
delimiters. Default event and result serialization omits them from the public
transcript. The model still receives their full content and tags.

To show summaries in a client, have the backend select them with
`is_summary_item()` from authoritative run history or an `AgentSnapshot`, then
send the selected content to the client. Both sources retain the complete
summary and its internal metadata. Default public serializers do not expose
this content. For summaries without metadata, the opening and closing tags
must occupy the first and last lines, with optional surrounding whitespace.
Ordinary user text that quotes the tags is not a summary.

Use `render_summary_item(bridge=..., summary_text=...)` to create the history
item, and `is_summary_item(item)` to detect an existing summary during later
compaction passes.

The default summary item template is Jinja2-rendered and exported as
`DEFAULT_SUMMARY_ITEM_TEMPLATE`. The summary markers are also exported for
advanced integrations that need to inspect the raw content.

## Custom Compactors

A custom compactor implements `async compact(request: CompactionRequest)` and
returns a `CompactionResult`. It owns final assembly, including
`request.required_messages` and `request.preserved_context`. A changed result
must contain every required message. The shim rejects an omitted anchor and
does not append it afterward. Custom compactors also own the handling of state
projections; the shim does not restore omitted plan or application context.

Prefer a preservation selector when only retention policy differs. To keep
native assembly in a custom compactor, delegate to `DefaultCompactor`:

```python
from agentlane.harness.compaction import (
    CompactionRequest,
    CompactionResult,
    DefaultCompactor,
    DefaultCompactorConfig,
)


class ApplicationCompactor:
    def __init__(self, summary_prompt: str) -> None:
        self._native = DefaultCompactor(
            DefaultCompactorConfig(prompt=summary_prompt)
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return await self._native.compact(request)
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

All `prepare_turn(...)` callbacks finish before `prepare_request(...)` begins.
Owners publish current state through `turn.preserved_context` during turn
preparation. `CompactionShim.prepare_request(...)` then selects and summarizes
against those contributions. Final request validation runs after every
`transform_messages(...)` callback. Context appended by a later transform is
checked by final validation but is outside the pre-summary selection.

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
