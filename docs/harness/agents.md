# Harness Agents

Date: 2026-04-02
Status: Phase 3 baseline

## What The Default Agent Owns

The default harness `Agent` builds on `Task` and owns:

1. agent identity and descriptive metadata,
2. system-prompt seeding for new conversations,
3. canonical conversation history,
4. queued inbound user turns, and
5. delegation of each turn to a stateless `Runner`.

## Runtime Guarantee Preserved

The harness preserves the current runtime rule that only one handler for a given
`AgentId` is active at a time.

That means "message injection while running" is modeled as:

1. append the new user turn to the agent's internal queue,
2. finish the currently running runner turn, and then
3. process the queued message on the next loop turn before the agent becomes idle.

The harness does not introduce concurrent re-entry for the same `AgentId`.

## Drain Strategy

Queued user turns are drained one runner turn at a time.

That means:

1. the lifecycle pops one queued user turn,
2. appends only that one user turn to the conversation history,
3. invokes the runner once, and then
4. repeats for the next queued turn if any remain.

The lifecycle does not batch all outstanding queued user turns into one larger runner call.

## Conversation Lifecycle

### New `AgentId`

When a new agent instance receives its first user message:

1. the conversation history is initialized,
2. the system prompt is added first if configured, and
3. the inbound message is appended as a canonical user-role message.

### Existing idle `AgentId`

When an existing idle agent receives another user message:

1. the prior message history is preserved,
2. the new user turn is appended, and
3. the runner is invoked again with the full conversation state.

### Existing running `AgentId`

When a user message arrives while the agent is already executing a runner turn:

1. the message is queued internally, and
2. it is processed on the next turn immediately after the active turn completes.

## Message Shape

The public runtime payload is `UserMessage`.

For convenience, `Agent.user_message(content)` returns one `UserMessage`.

Internal turn queueing stays internal to the agent lifecycle. The `_enqueue_user_message`
method is a private harness seam used by lifecycle tests and by the agent's own handler.

Inbound content is rendered into canonical `MessageDict` form:

1. string content stays as string content,
2. list content is preserved for multipart inputs, and
3. other content is serialized to JSON when possible, falling back to `str(...)`.

This message construction now lives in `agentlane.models`, not in the harness lifecycle layer.

## Phase Boundary

Phase 3 stops at lifecycle and queueing semantics.

1. The runner contract remains stateless and externally provided.
2. Tool execution behavior is not implemented here.
3. Handoffs and sub-agent delegation are not implemented here.
