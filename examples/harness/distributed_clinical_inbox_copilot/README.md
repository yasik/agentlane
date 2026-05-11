# Distributed Clinical Inbox Copilot

This demo combines a streamed top-level harness agent with an explicit
distributed runtime topology for a clinical inbox scenario.

It demonstrates:

1. interactive clinician input through the terminal
2. `DefaultAgent.run_stream(...)` for top-level streaming
3. visible model reasoning summaries, provider phases, and tool-call arguments
4. one `WorkerAgentRuntimeHost` plus dedicated `WorkerAgentRuntime` workers
5. resolved host:port and worker id for each runtime node in the UI
6. distributed `publish_message(...)` fan-out to parallel specialist agents
7. stateful distributed fan-in through an aggregator worker

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/distributed_clinical_inbox_copilot/main.py
```

You can also supply inputs non-interactively:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/distributed_clinical_inbox_copilot/main.py \
  --clinician-name "Dr. Rivera" \
  --patient-label "Maya R., 54F" \
  --patient-message "I started the new injection and now I feel dizzy. My sugar was 64 this morning and 68 after lunch. What should I do?"
```

## Topology

The demo starts one host and six workers:

1. `copilot-worker` runs the streamed top-level harness agent
2. `med-safety-worker` handles medication-safety review requests
3. `guideline-worker` handles guideline-style escalation review
4. `chart-history-worker` extracts relevant chart context
5. `patient-comms-worker` drafts patient communication guidance
6. `aggregator-worker` collects specialist findings by review id

The live dashboard shows the host address, each worker host:port, and each
worker id. Specialist work is published through the host using
`REVIEW_TOPIC_TYPE`; each specialist publishes a finding to `RESULT_TOPIC_TYPE`;
the stateful aggregator completes the model-facing tool result after all
findings arrive.
