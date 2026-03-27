# agentlane-openai

`agentlane-openai` adapts the OpenAI Responses API to AgentLane's unified model interfaces.

This package exists so AgentLane can use the OpenAI SDK directly while still exposing the same core model boundary used elsewhere in the framework. The shared contracts live in `agentlane.models`; this package implements those contracts with a native OpenAI client.

The main public entrypoints are:

1. `ResponsesClient`
2. `ResponsesFactory`
3. `EmbeddingsClient`
4. `ResponsesApiOutputAdapter`

Use this package when you want first-party OpenAI Responses API behavior behind AgentLane's common client interface, or when you need the small embeddings wrapper that lives alongside that integration.
