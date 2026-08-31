# agentlane-claude-agent-sdk

`agentlane-claude-agent-sdk` contains the optional AgentLane integration for the Claude Agent SDK.

Install it with `agentlane[claude-agent-sdk]`. A base AgentLane installation does not install the Claude Agent SDK.

The integration accepts one text message. It starts a new SDK session for that message and returns the final text result.

Import `ClaudeAgent` from `agentlane_claude_agent_sdk`. Bind it to an AgentLane runtime like any other `Task`:

```python
from agentlane_claude_agent_sdk import ClaudeAgent

ClaudeAgent.bind(runtime, claude_agent_id)
```

Without explicit SDK options, the task disables tools, skills, settings sources, and MCP servers. It also limits the query to one turn. You can pass a `ClaudeAgentOptions` instance through the `options` argument. The task rejects options that continue, resume, or fork a session.

The task drains the SDK stream and returns only the last successful string result. SDK errors and invalid terminal results become AgentLane handler errors.
