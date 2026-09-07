# agentlane-openai

`agentlane-openai` adapts the OpenAI Responses API to AgentLane's unified model interfaces.

This package exists so AgentLane can use the OpenAI SDK directly while still exposing the same core model boundary used elsewhere in the framework. The shared contracts live in `agentlane.models`; this package implements those contracts with a native OpenAI client.

The main public entrypoints are:

1. `ResponsesClient`
2. `ResponsesFactory`
3. `EmbeddingsClient`
4. `ResponsesApiOutputAdapter`

Use this package when you want first-party OpenAI Responses API behavior behind AgentLane's common client interface, or when you need the small embeddings wrapper that lives alongside that integration.

`ResponsesClient` accepts conversation history in Chat Completions message
format. Developer instructions can be strings or text-part lists. Assistant
history retains nonempty text parts in order, refusal content, and a supplied
`phase` on message items. Replayed assistant messages use `status: completed`
unless the caller supplies a status. A supplied message ID is retained; the
converter does not create IDs. Empty assistant text parts are omitted, and
tool-only turns do not produce an empty assistant message.

Assistant content stays before its function calls, followed by tool results
in conversation order. This applies to OpenAI and Azure, with and without
streaming. System messages are sent with the developer role.

Supply the required history on each call when you manage conversation state
yourself. The client does not automatically chain response IDs. See OpenAI's
[conversation state guide](https://developers.openai.com/api/docs/guides/conversation-state).
The message converter does not accept a complete native Responses output
array as a replacement for Chat Completions messages.
