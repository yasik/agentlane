# Persistent Agent Quickstart

This is the shortest path to an agent that survives process restarts. One JSON
file owns the agent address, rendered instructions, conversation, harness
state, and committed revision.

Run the script twice with the same state path:

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/persistent_agent_quickstart/main.py \
  "Remember that my portfolio risk limit is 8%."
uv run python examples/harness/persistent_agent_quickstart/main.py \
  "What is my portfolio risk limit?"
```

The second process restores the first conversation before appending its new
prompt. Each successful run atomically replaces
`.agentlane/persistent-assistant.json` with the next revision.

Use `--state another/path.json` to choose another agent state. Remove the file
or call `agent.reset()` to start over.
