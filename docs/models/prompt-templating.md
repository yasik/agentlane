# Models: Prompt Templating

Prompt templating is the part of AgentLane that turns typed application data
into the messages sent to a model. It gives you a place to describe the shape
of your instructions once, then render those instructions with concrete values
for a specific run.

The main building blocks are
[`PromptTemplate`](../../src/agentlane/models/_prompts.py) for text prompts,
[`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py) for
structured prompt content, and
[`PromptSpec`](../../src/agentlane/models/_prompts.py) for pairing a template
with the values used to render it. Multipart prompts use
[`TextPart`](../../src/agentlane/models/_prompts.py),
[`FilePart`](../../src/agentlane/models/_prompts.py), and
[`ImagePart`](../../src/agentlane/models/_prompts.py), while
[`OutputSchema`](../../src/agentlane/models/_output_schema.py) describes the
expected model output.

## Core Ideas

The templating surface is intentionally small:

1. define a template once
2. supply typed values later
3. render those values into canonical model messages when the run executes

That separation keeps prompt structure reusable while letting each run provide
its own input values.

## `PromptTemplate`

Use [`PromptTemplate`](../../src/agentlane/models/_prompts.py) when your prompt
is plain text and naturally fits into optional `system` and `user` messages.

It accepts:

1. `system_template`
2. `user_template`
3. `output_schema`

At least one of `system_template` or `user_template` must be present.

Template strings are rendered with Jinja. In practice that means you can define
variables once and pass a typed object, often a `TypedDict`, when it is time to
render the prompt.

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

## `PromptSpec`

[`PromptSpec`](../../src/agentlane/models/_prompts.py) is the developer-facing
container for one rendered prompt input. It keeps the template and the concrete
values together so the harness can build model messages later.

This is useful when:

1. the same template is reused across runs
2. you want type-checked prompt values
3. the prompt should stay at the higher-level harness boundary instead of being
   flattened into raw message dictionaries by application code

## Multipart Prompts

Use [`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py) when
message content needs more than text. This is the right surface when the model
input should include files, images, or mixed content blocks.

Multipart prompts are made from part templates:

1. [`TextPart`](../../src/agentlane/models/_prompts.py) for rendered text
2. [`FilePart`](../../src/agentlane/models/_prompts.py) for base64 file input
3. [`ImagePart`](../../src/agentlane/models/_prompts.py) for base64 image input

`TextPart` renders Jinja templates with context values. `FilePart` and
`ImagePart` carry already-prepared base64 payloads.

Use multipart prompts when the model input itself is structured. Use
`PromptTemplate` when text alone is enough.

## Output Schemas And Prompting

Every prompt template carries an
[`OutputSchema`](../../src/agentlane/models/_output_schema.py). That schema is
how the model layer knows whether you expect:

1. plain text
2. structured JSON output
3. strict schema enforcement where the provider supports it

In other words, prompt templating describes the input shape and `OutputSchema`
describes the expected output shape.

## Rendering Model Messages

Templates render into canonical message payloads through `render_messages(...)`.
Most application code does not need to call that method directly. The harness
can accept a [`PromptSpec`](../../src/agentlane/models/_prompts.py) and build
the final model request for you.

Reach for direct rendering when you are implementing lower-level model flows.
For normal harness usage, keep prompts at the `PromptSpec` level.

## Related Docs

1. [Models Overview](./overview.md)
2. [Harness Architecture](../harness/architecture.md)
3. [Harness Runner](../harness/runner.md)
