# agentlane-braintrust

`agentlane-braintrust` provides the Braintrust tracing integration for AgentLane. It connects AgentLane traces and spans to Braintrust.

The public surface is intentionally small:

1. `BraintrustProcessor`

Use this package when you want to export AgentLane tracing data to Braintrust. If you need traces, spans, metrics, or context propagation themselves, use `agentlane.tracing` from the core package instead.
