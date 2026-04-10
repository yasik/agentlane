# Transport Serialization

This page explains how AgentLane turns Python values into wire-safe payloads.
Most application code can ignore these details because the defaults cover the
common cases. The transport layer matters when messages cross process
boundaries, when types need stable schema identifiers, or when you want custom
serialization behavior.
The main transport types are
[`SerializerRegistry`](../../src/agentlane/transport/_registry.py), which keeps
track of serializer selection and type metadata,
[`MessageSerializer`](../../src/agentlane/transport/_serializer.py), which
defines the serializer contract, and
[`WirePayload`](../../src/agentlane/transport/_wire_payload.py), which is the
transport-safe payload shape exchanged across boundaries.

## TL;DR

Default path requires no serializer setup:

```python
runtime = SingleThreadedRuntimeEngine()
result = await runtime.send_message(MyMessage(...), recipient)
```

Manual registry/serializer wiring is optional and only for advanced customization.

## Key Principles

1. Serialization is a transport concern, not core routing logic.
2. Runtime defaults should work out of the box for common payload types.
3. Explicit serializer wiring is optional and only for advanced customization.
4. Schema identifiers remain globally namespaced strings.

## Default Behavior (Recommended)

Every runtime owns a default
[`SerializerRegistry`](../../src/agentlane/transport/_registry.py) with
auto-inference enabled.
For dataclass, pydantic, protobuf, and generic JSON-compatible payloads, no manual serializer registration is required.

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

The runtime/transport layer infers and caches:

1. `schema_id` from Python type (`module.qualname`, normalized).
2. `content_type` from payload kind.
3. concrete serializer implementation for that `(schema_id, content_type)`.

Note:
Typed decode for payloads that arrive without prior local registration (for example remote protobuf payloads) still requires explicit serializer registration. JSON payloads can fall back to plain `dict`/`list` values.

## Optional Explicit Configuration

Use explicit configuration only when you need strict schema contracts or non-default content types.

```python
from pydantic import BaseModel

from agentlane.transport import SerializerRegistry


class TaskModel(BaseModel):
    name: str


registry = SerializerRegistry()
registry.register_type(TaskModel)
```

Manual [`MessageSerializer`](../../src/agentlane/transport/_serializer.py)
implementations are still supported as an escape hatch.

## Boundary Conversion

Use runtime hooks (or transport helpers) to convert between messaging payloads and wire payloads:

```python
wire_payload = runtime.payload_to_wire_payload(payload)
restored_payload = runtime.wire_payload_to_payload(wire_payload)
```

## Migration Note

Implicit map-based metadata (`attributes`) remains intentionally excluded from serialization contracts.
Use typed fields (for example `idempotency_key`) and first-class message schemas.
