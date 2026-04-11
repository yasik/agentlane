# Transport Serialization

Most AgentLane code does not need to think about serialization at all. If you
send dataclasses, Pydantic models, protobuf messages, or plain JSON-compatible
values, the runtime usually has enough information to do the right thing.

Transport becomes important when work crosses a process boundary, when you need
stable schema identifiers, or when you want to take control of how values are
encoded and decoded.

When that boundary matters, the runtime leans on
[`SerializerRegistry`](../../src/agentlane/transport/_registry.py) to choose a
[`MessageSerializer`](../../src/agentlane/transport/_serializer.py), and the
encoded value travels as a
[`WirePayload`](../../src/agentlane/transport/_wire_payload.py).

## The Default Path

Every runtime owns a default
[`SerializerRegistry`](../../src/agentlane/transport/_registry.py). That
registry can infer a serializer from the Python value you send.

In the common case, the runtime figures out:

1. a schema id for the value's type
2. the content type for the payload
3. which serializer should handle that combination

That means most application code can simply send a value and let the transport
layer do the rest.

```python
from pydantic import BaseModel

from agentlane.messaging import AgentId, AgentType
from agentlane.runtime import SingleThreadedRuntimeEngine


class TaskModel(BaseModel):
    name: str


runtime = SingleThreadedRuntimeEngine()
recipient = AgentId(type=AgentType("planner"), key="default")
result = await runtime.send_message(TaskModel(name="compile"), recipient)
```

## When Explicit Registration Helps

Explicit registration is useful when you need stricter control than the default
inference path gives you.

Typical reasons include:

1. remote decode should reconstruct a typed value instead of falling back to a
   plain dict or list
2. a custom content type or serializer is part of your contract
3. the payload type is uncommon enough that you do not want to rely on inference

```python
from pydantic import BaseModel

from agentlane.transport import SerializerRegistry


class TaskModel(BaseModel):
    name: str


registry = SerializerRegistry()
registry.register_type(TaskModel)
```

If you need full control, implement the
[`MessageSerializer`](../../src/agentlane/transport/_serializer.py) protocol
directly and register that serializer yourself.

## Wire Payloads

At the transport boundary, runtime payloads become
[`WirePayload`](../../src/agentlane/transport/_wire_payload.py) values. Most
code never needs to construct them directly, but the runtime exposes helpers for
cases where you do want to convert a payload manually.

```python
wire_payload = runtime.payload_to_wire_payload(payload)
restored_payload = runtime.wire_payload_to_payload(wire_payload)
```

## A Useful Rule Of Thumb

If both sender and receiver live inside one normal runtime and you are sending
ordinary Python models, do not over-configure serialization. Start with the
default path and only reach for explicit registry setup when you have a clear
transport requirement.
