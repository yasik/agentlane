# Models: Prompt Templating

Prompt code gets hard to maintain when message strings, runtime data, and
expected output shape are all mixed together. AgentLane keeps those concerns
separate so a prompt can stay readable while each run still provides its own
values.

Most prompt flows use
[`PromptTemplate`](../../src/agentlane/models/_prompts.py) for text messages or
[`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py) when a
message needs multiple content parts. [`PromptSpec`](../../src/agentlane/models/_prompts.py)
is the small object that binds one of those templates to the values for the
current run.

## A Template Is Structure Plus Values

The usual flow is:

1. define a template once
2. pass values later
3. let the harness or model layer render the final messages

That is what [`PromptSpec`](../../src/agentlane/models/_prompts.py) is for. It
keeps the prompt definition and the current values together so the rest of the
framework can build a request at the right time.

## Text Prompts

Use [`PromptTemplate`](../../src/agentlane/models/_prompts.py) when your prompt
fits naturally into optional `system` and `user` messages.

This is the most common case. Template strings are rendered with Jinja, so you
can keep the prompt readable and still pass typed values into it.

```python
from typing import TypedDict

from agentlane.models import OutputSchema, PromptSpec, PromptTemplate


class SupportVars(TypedDict):
    customer_tier: str
    question: str


template = PromptTemplate[SupportVars, str](
    system_template="You are helping a {{ customer_tier }} customer.",
    user_template="{{ question }}",
    output_schema=OutputSchema(str),
)

prompt = PromptSpec(
    template=template,
    values={
        "customer_tier": "business",
        "question": "How do I rotate API keys?",
    },
)
```

## Multipart Prompts

Use [`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py) when
the model input needs more than plain text.

That usually means one of two things:

1. the message content should be built from multiple parts
2. the prompt should include file or image input alongside text

Multipart templates use
[`TextPart`](../../src/agentlane/models/_prompts.py),
[`FilePart`](../../src/agentlane/models/_prompts.py), and
[`ImagePart`](../../src/agentlane/models/_prompts.py). `TextPart` renders a
template string with values. `FilePart` and `ImagePart` carry already-prepared
payloads.

```python
from agentlane.models import ImagePart, MultiPartPromptTemplate, OutputSchema
from agentlane.models import PromptSpec, TextPart


receipt_image_b64 = "..."

template = MultiPartPromptTemplate[dict[str, str], str](
    system_parts=[TextPart("Summarize the issue shown in the image.")],
    user_parts=[
        TextPart("Customer note: {{ note }}"),
        ImagePart(base64_data=receipt_image_b64, media_type="image/png"),
    ],
    output_schema=OutputSchema(str),
)

prompt = PromptSpec(
    template=template,
    values={"note": "The package arrived damaged."},
)
```

## How `OutputSchema` Fits

Every prompt template carries an
[`OutputSchema`](../../src/agentlane/models/_output_schema.py). That schema is
how the model layer knows whether you expect plain text or structured output,
and whether a provider should enforce a JSON schema at request time.

It helps to think of the pieces this way:

1. the template describes what goes in
2. the output schema describes what should come back

## When To Render Directly

Templates ultimately render through `render_messages(...)`, but most
application code does not need to call that method itself.

If you are building on the harness, keep prompts at the
[`PromptSpec`](../../src/agentlane/models/_prompts.py) level and let the runner
build the final request. Reach for direct rendering when you are implementing a
lower-level model flow and need the canonical message list immediately.

## Related Docs

1. [Models Overview](./overview.md)
2. [Harness Architecture](../harness/architecture.md)
3. [Harness Runner](../harness/runner.md)
