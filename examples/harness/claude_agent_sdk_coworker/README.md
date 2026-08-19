# Claude Agent SDK Coworker

This example proves that a native AgentLane agent can send an addressed task to
a Claude Agent SDK participant, receive its text result, and complete the
original AgentLane run.

## Run

Run these commands from the repository root:

```bash
uv sync --extra claude-agent-sdk
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
uv run python examples/harness/claude_agent_sdk_coworker/main.py
```

The example prints three labeled stages:

1. The native AgentLane identity sends a task to the fixed Claude identity.
2. The Claude result returns to the native agent through the delivery result
   and tool result.
3. The native AgentLane agent uses that result to write the final answer.

The shared runtime uses `worker_count=2`. One worker runs the native agent while
it waits for its tool. The second worker handles the nested Claude delivery. A
one-worker runtime cannot make progress in this topology.

## Data and credentials

This example sends prompts and results to both OpenAI and Anthropic. Use only
synthetic or non-sensitive data.

The Claude Agent SDK starts its bundled CLI as a child process. That process
inherits the parent process environment. AgentLane does not isolate secrets for
this example. Start it with only the environment variables that the two model
providers need. Do not run it in an environment that contains unrelated
secrets.

## Proof boundary

This proof shows outbound AgentLane addressing: the Claude participant receives
the fixed native AgentLane sender identity. Claude's answer returns through the
same delivery call and becomes a native model tool result. The example does not
send a second addressed message from Claude to the native agent.

This POC does not prove Claude session continuity, tool translation, streaming
translation, permission-policy parity, or cancellation parity. Each addressed
Claude task starts one fresh, tool-free SDK query.
