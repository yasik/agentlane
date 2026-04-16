# Clinical Inbox Copilot

This announcement-oriented demo combines a streamed top-level harness agent with
runtime-driven parallel specialist review for a clinical inbox scenario.

It demonstrates:

1. interactive clinician input through the terminal
2. `DefaultAgent.run_stream(...)` for top-level streaming
3. visible model reasoning summaries, provider phases, and tool-call arguments
4. one runtime `publish_message(...)` fan-out to parallel specialist agents
5. one stateful aggregator that merges specialist findings back into the tool result
6. polished `rich` console output suitable for VHS capture

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/clinical_inbox_copilot/main.py
```

You can also supply inputs non-interactively:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/clinical_inbox_copilot/main.py \
  --clinician-name "Dr. Rivera" \
  --patient-label "Maya R., 54F" \
  --patient-message "I started the new injection and now I feel dizzy. My sugar was 64 this morning and 68 after lunch. What should I do?"
```
