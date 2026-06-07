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

1. a schema id for the value's type (`infer_schema_id_for_value` /
   `infer_schema_id_for_type`)
2. the content type for the payload (`infer_content_type_for_value` /
   `infer_content_type_for_type`)
3. which serializer should handle that combination

That means most application code can simply send a value and let the transport
layer do the rest. The runtime's default registry is produced by
[`create_default_serializer_registry`](../../src/agentlane/transport/_registry.py),
which constructs a `SerializerRegistry(auto_register_defaults=True)`.

`SerializerRegistry` exposes two operating modes through its
`auto_register_defaults` flag:

- default mode (`auto_register_defaults=True`): the registry infers and caches a
  serializer for a common payload type the first time it is used
- strict mode (`auto_register_defaults=False`): the serializer key must already
  be registered or encode/decode raises `UnknownSerializerError`

Beyond `register_type`, the registry surface includes `register`,
`register_many`, `unregister`, `has`, `encode`, `decode`, and the read-only
`serializers` snapshot.

```python
from pydantic import BaseModel

from agentlane.messaging import AgentId
from agentlane.runtime import SingleThreadedRuntimeEngine


class TaskModel(BaseModel):
    name: str


runtime = SingleThreadedRuntimeEngine()
recipient = AgentId.from_values("planner", "default")
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

## Built-in Serializers

The registry ships four concrete serializers, all keyword-only. Default
inference (and `register_type`) picks one of them based on the Python type:

- [`PydanticJsonSerializer`](../../src/agentlane/transport/_codecs_json.py)
  (`*, schema_id, model_type, content_type=JSON_CONTENT_TYPE`) for
  `pydantic.BaseModel` subclasses.
- [`DataclassJsonSerializer`](../../src/agentlane/transport/_codecs_json.py)
  (`*, schema_id, model_type, content_type=JSON_CONTENT_TYPE`) for dataclass
  types.
- [`ProtobufSerializer`](../../src/agentlane/transport/_codecs_protobuf.py)
  (`*, schema_id, message_type, content_type=PROTOBUF_CONTENT_TYPE`) for
  `google.protobuf.message.Message` subclasses.
- [`JsonValueSerializer`](../../src/agentlane/transport/_codecs_json.py)
  (`*, schema_id, content_type=JSON_CONTENT_TYPE`) for any other JSON-compatible
  value. This is also the conservative decode-only fallback: a JSON payload with
  no registered typed serializer is recovered as plain `dict`/`list`/scalars,
  while typed protobuf, Pydantic, and dataclass decoding always requires
  explicit registration.

## Wire Payloads

At the transport boundary, runtime payloads become
[`WirePayload`](../../src/agentlane/transport/_wire_payload.py) values. Most
code never needs to construct them directly, but the runtime exposes helpers for
cases where you do want to convert a payload manually.

```python
wire_payload = runtime.payload_to_wire_payload(payload)
restored_payload = runtime.wire_payload_to_payload(wire_payload)
```

A [`WirePayload`](../../src/agentlane/transport/_wire_payload.py) carries four
fields: `schema_id` ([`SchemaId`](../../src/agentlane/transport/_types.py)),
`content_type` ([`ContentType`](../../src/agentlane/transport/_types.py)),
`encoding` ([`WireEncoding`](../../src/agentlane/transport/_types.py): `JSON`,
`PROTOBUF`, or `BYTES`), and `body` (the serialized `bytes`).

These transport identity types validate eagerly:

- `SchemaId` must be non-empty, use only letters, numbers, and `_.:/-`
  separators, and be globally namespaced — it must contain at least one of `.`,
  `:`, or `/`, or construction raises `ValueError`.
- `ContentType` must be MIME-like (for example `application/json`), or
  construction raises `ValueError`.

The canonical content-type constants are `JSON_CONTENT_TYPE`
(`application/json`), `PROTOBUF_CONTENT_TYPE` (`application/x-protobuf`), and
`OCTET_STREAM_CONTENT_TYPE` (`application/octet-stream`).

When converting between the transport and messaging layers, `WireEncoding` maps
to the messaging `PayloadFormat` through the
[`wire_encoding_for_payload_format`](../../src/agentlane/transport/_wire_payload.py)
and
[`payload_format_for_wire_encoding`](../../src/agentlane/transport/_wire_payload.py)
helpers.

## Errors

All transport serializer errors subclass
[`SerializationError`](../../src/agentlane/transport/_errors.py):

- `SerializerConflictError` — raised by `register` (and therefore
  `register_many` / `register_type`) when a serializer already exists for the
  `(schema_id, content_type)` key and `replace=False`.
- `UnknownSerializerError` — raised by `unregister` when the key is absent, and
  by `encode`/`decode` when no serializer resolves for the key.
- `SerializerEncodeError` — raised by `encode` when the underlying codec fails
  during object-to-bytes conversion.
- `SerializerDecodeError` — raised by `decode` when the underlying codec fails
  during bytes-to-object conversion.

## A Useful Rule Of Thumb

If both sender and receiver live inside one normal runtime and you are sending
ordinary Python models, do not over-configure serialization. Start with the
default path and only reach for explicit registry setup when you have a clear
transport requirement.
