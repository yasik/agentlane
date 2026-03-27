# agentlane-litellm

`agentlane-litellm` adapts LiteLLM to AgentLane's unified model interfaces.

This package exists so AgentLane can use LiteLLM as a provider backend without leaking LiteLLM-specific request and client details into the framework core. The core contracts live in `agentlane.models`; this package implements them against LiteLLM.

The main public entrypoints are:

1. `Client`
2. `Factory`

Use this package when you want a generic provider-backed model client that conforms to AgentLane's shared `Model` and `Factory` interfaces while still routing calls through LiteLLM.
