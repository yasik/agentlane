# Harness Examples

This category is reserved for runnable examples that exercise the agentic
harness directly.

## Available demos

1. [default_agent_quickstart](./default_agent_quickstart/): highest-level local quickstart using `agentlane.harness.agents.DefaultAgent` with no manual runtime or runner wiring.
2. [default_agent_shims_quickstart](./default_agent_shims_quickstart/): custom harness shim quickstart showing instruction augmentation and persisted `RunState.shim_state`.
3. [default_agent_streaming_quickstart](./default_agent_streaming_quickstart/): highest-level local streaming quickstart using `DefaultAgent.run_stream(...)`, including provider-native reasoning and preamble details.
4. [default_agent_skills_quickstart](./default_agent_skills_quickstart/): local skills quickstart using `SkillsShim` plus an explicit `FilesystemSkillLoader`.
5. [base_tools_quickstart](./base_tools_quickstart/): first-party base tools quickstart exposing `read`, `find`, `grep`, and `write` through `HarnessToolsShim`.
6. [base_tools_plan_quickstart](./base_tools_plan_quickstart/): focused `plan` tool quickstart showing replacement semantics and persisted shim state.
7. [clinical_inbox_copilot](./clinical_inbox_copilot/): clinical case triage demo with skills and lifecycle hooks.
8. [streaming_escalation_flow](./streaming_escalation_flow/): one streamed run that combines a normal tool call, a predefined agent-as-tool delegation, and a first-class handoff.
9. [customer_support_conversation](./customer_support_conversation/): real OpenAI-backed multi-turn support conversation with templated instructions and `RunState` resume.
10. [tool_calling_search_answer](./tool_calling_search_answer/): real OpenAI-backed single-question demo showing the runner-owned tool loop with one mocked search result.
11. [agent_as_tool_policy_specialist](./agent_as_tool_policy_specialist/): predefined agent-as-tool demo where a manager delegates a warranty question to a child policy specialist and then resumes.
12. [default_agent_tool_note_writer](./default_agent_tool_note_writer/): generic `agent` tool demo where the model chooses a helper name, optional description, and a focused task.
13. [handoff_to_returns_specialist](./handoff_to_returns_specialist/): predefined first-class handoff demo where support triage transfers the conversation to a returns specialist.
14. [default_handoff_takeover](./default_handoff_takeover/): generic `handoff` demo where triage transfers the conversation to a fresh specialist created from `DefaultHandoff(...)`.

## Run

All harness demos require `OPENAI_API_KEY` in the environment:

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/clinical_inbox_copilot/main.py
uv run python examples/harness/default_agent_quickstart/main.py
uv run python examples/harness/default_agent_shims_quickstart/main.py
uv run python examples/harness/default_agent_streaming_quickstart/main.py
uv run python examples/harness/default_agent_skills_quickstart/main.py
uv run python examples/harness/base_tools_quickstart/main.py
uv run python examples/harness/streaming_escalation_flow/main.py
uv run python examples/harness/customer_support_conversation/main.py
uv run python examples/harness/tool_calling_search_answer/main.py
uv run python examples/harness/agent_as_tool_policy_specialist/main.py
uv run python examples/harness/default_agent_tool_note_writer/main.py
uv run python examples/harness/handoff_to_returns_specialist/main.py
uv run python examples/harness/default_handoff_takeover/main.py
uv run python examples/harness/base_tools_plan_quickstart/main.py
```
