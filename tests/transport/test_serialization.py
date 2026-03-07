from dataclasses import dataclass
from threading import Thread

import pytest
from google.protobuf.wrappers_pb2 import StringValue
from pydantic import BaseModel

from agentlane.messaging import Payload, PayloadFormat
from agentlane.runtime import RuntimeEngine, SingleThreadedRuntimeEngine
from agentlane.transport import (
    ContentType,
    DataclassJsonSerializer,
    ProtobufSerializer,
    PydanticJsonSerializer,
    SchemaId,
    SerializerConflictError,
    SerializerRegistry,
    UnknownSerializerError,
    WireEncoding,
    WirePayload,
    infer_content_type_for_value,
    infer_schema_id_for_value,
    payload_to_wire_payload,
    wire_payload_to_payload,
)


@dataclass(slots=True)
class TaskEnvelope:
    name: str


class TaskModel(BaseModel):
    name: str


def test_schema_id_requires_global_namespace() -> None:
    with pytest.raises(ValueError):
        _ = SchemaId("TaskCreated")

    schema_id = SchemaId("agentlane.workflow.task.created.v1")
    assert schema_id.value == "agentlane.workflow.task.created.v1"


def test_content_type_requires_mime_like_shape() -> None:
    with pytest.raises(ValueError):
        _ = ContentType("application")

    content_type = ContentType("application/json")
    assert content_type.value == "application/json"


def test_registry_detects_conflicts_for_same_key() -> None:
    registry = SerializerRegistry()
    serializer = DataclassJsonSerializer(
        schema_id="agentlane.workflow.task.envelope.v1",
        model_type=TaskEnvelope,
    )
    registry.register(serializer)

    with pytest.raises(SerializerConflictError):
        registry.register(serializer)


def test_registry_register_type_is_concise_and_works() -> None:
    registry = SerializerRegistry()
    registered = registry.register_type(TaskModel)

    wire_payload = registry.encode(
        TaskModel(name="compile"),
        schema_id=registered.schema_id,
        content_type=registered.content_type,
    )
    decoded = registry.decode(wire_payload)

    assert decoded == TaskModel(name="compile")


def test_registry_encodes_and_decodes_pydantic_models() -> None:
    registry = SerializerRegistry()
    schema_id = SchemaId("agentlane.workflow.task.model.v1")
    serializer = PydanticJsonSerializer(
        schema_id=schema_id,
        model_type=TaskModel,
    )
    registry.register(serializer)

    wire_payload = registry.encode(
        TaskModel(name="compile"),
        schema_id=schema_id,
        content_type="application/json",
    )
    decoded = registry.decode(wire_payload)

    assert wire_payload.schema_id == schema_id
    assert wire_payload.encoding == WireEncoding.JSON
    assert decoded == TaskModel(name="compile")


def test_registry_auto_registers_common_type_on_encode() -> None:
    registry = SerializerRegistry()
    schema_id = infer_schema_id_for_value(TaskModel(name="compile"))
    content_type = infer_content_type_for_value(TaskModel(name="compile"))

    wire_payload = registry.encode(
        TaskModel(name="compile"),
        schema_id=schema_id,
        content_type=content_type,
    )
    decoded = registry.decode(wire_payload)

    assert wire_payload.encoding == WireEncoding.JSON
    assert decoded == TaskModel(name="compile")


def test_registry_decode_json_fallback_returns_plain_value() -> None:
    registry = SerializerRegistry()
    wire_payload = WirePayload(
        schema_id=SchemaId("agentlane.workflow.untyped.json.v1"),
        content_type=ContentType("application/json"),
        encoding=WireEncoding.JSON,
        body=b'{"name":"compile"}',
    )

    decoded = registry.decode(wire_payload)

    assert decoded == {"name": "compile"}


def test_registry_encodes_and_decodes_dataclass_models() -> None:
    registry = SerializerRegistry()
    schema_id = SchemaId("agentlane.workflow.task.envelope.v1")
    serializer = DataclassJsonSerializer(
        schema_id=schema_id,
        model_type=TaskEnvelope,
    )
    registry.register(serializer)

    wire_payload = registry.encode(
        TaskEnvelope(name="compile"),
        schema_id=schema_id,
        content_type="application/json",
    )
    decoded = registry.decode(wire_payload)

    assert wire_payload.schema_id == schema_id
    assert wire_payload.encoding == WireEncoding.JSON
    assert decoded == TaskEnvelope(name="compile")


def test_registry_encodes_and_decodes_protobuf_models() -> None:
    registry = SerializerRegistry()
    schema_id = SchemaId("agentlane.workflow.protobuf.string_value.v1")
    serializer = ProtobufSerializer(
        schema_id=schema_id,
        message_type=StringValue,
    )
    registry.register(serializer)

    wire_payload = registry.encode(
        StringValue(value="compile"),
        schema_id=schema_id,
        content_type="application/x-protobuf",
    )
    decoded = registry.decode(wire_payload)

    assert wire_payload.schema_id == schema_id
    assert wire_payload.encoding == WireEncoding.PROTOBUF
    assert isinstance(decoded, StringValue)
    assert decoded.value == "compile"


def test_registry_fails_fast_for_unknown_serializer_keys_when_auto_disabled() -> None:
    registry = SerializerRegistry(auto_register_defaults=False)
    with pytest.raises(UnknownSerializerError):
        _ = registry.encode(
            TaskModel(name="compile"),
            schema_id="agentlane.workflow.task.model.v1",
            content_type="application/json",
        )

    with pytest.raises(UnknownSerializerError):
        _ = registry.decode(
            WirePayload(
                schema_id=SchemaId("agentlane.workflow.task.model.v1"),
                content_type=ContentType("application/json"),
                encoding=WireEncoding.JSON,
                body=b'{"name":"compile"}',
            )
        )


def test_registry_rejects_incompatible_type_content_type_pair() -> None:
    registry = SerializerRegistry()

    with pytest.raises(UnknownSerializerError):
        _ = registry.encode(
            TaskModel(name="compile"),
            schema_id=SchemaId("agentlane.workflow.task.model.v1"),
            content_type=ContentType("application/x-protobuf"),
        )


def test_registry_decode_protobuf_requires_explicit_serializer() -> None:
    registry = SerializerRegistry()
    wire_payload = WirePayload(
        schema_id=SchemaId("agentlane.workflow.protobuf.string_value.v1"),
        content_type=ContentType("application/x-protobuf"),
        encoding=WireEncoding.PROTOBUF,
        body=StringValue(value="compile").SerializeToString(),
    )

    with pytest.raises(UnknownSerializerError):
        _ = registry.decode(wire_payload)


def test_boundary_conversion_roundtrip_uses_registry_codecs() -> None:
    registry = SerializerRegistry()
    schema_id = SchemaId("agentlane.workflow.task.model.v1")
    registry.register(
        PydanticJsonSerializer(
            schema_id=schema_id,
            model_type=TaskModel,
        )
    )

    payload = Payload(
        schema_name=schema_id.value,
        content_type="application/json",
        format=PayloadFormat.JSON,
        data=TaskModel(name="compile"),
    )
    wire_payload = payload_to_wire_payload(payload, registry=registry)
    restored_payload = wire_payload_to_payload(wire_payload, registry=registry)

    assert wire_payload.schema_id == schema_id
    assert restored_payload.format == PayloadFormat.JSON
    assert restored_payload.data == TaskModel(name="compile")


def test_runtime_boundary_hooks_auto_resolve_serializer() -> None:
    runtime: RuntimeEngine = SingleThreadedRuntimeEngine()
    payload = Payload(
        schema_name=infer_schema_id_for_value(TaskModel(name="compile")).value,
        content_type=infer_content_type_for_value(TaskModel(name="compile")).value,
        format=PayloadFormat.JSON,
        data=TaskModel(name="compile"),
    )

    wire_payload = runtime.payload_to_wire_payload(payload)
    restored_payload = runtime.wire_payload_to_payload(wire_payload)

    assert restored_payload.data == TaskModel(name="compile")
    assert restored_payload.format == PayloadFormat.JSON


def test_boundary_conversion_roundtrip_for_raw_bytes() -> None:
    registry = SerializerRegistry()
    payload = Payload(
        schema_name="agentlane.transport.raw.bytes.v1",
        content_type="application/octet-stream",
        format=PayloadFormat.BYTES,
        data=b"binary",
    )
    wire_payload = payload_to_wire_payload(payload, registry=registry)
    restored_payload = wire_payload_to_payload(wire_payload, registry=registry)

    assert wire_payload.encoding == WireEncoding.BYTES
    assert restored_payload.format == PayloadFormat.BYTES
    assert restored_payload.data == b"binary"


def test_runtime_owns_default_serializer_registry() -> None:
    runtime: RuntimeEngine = SingleThreadedRuntimeEngine()
    assert runtime.serializer_registry is not None


def test_runtime_allows_serializer_registry_override() -> None:
    serializer_registry = SerializerRegistry()
    runtime: RuntimeEngine = SingleThreadedRuntimeEngine(
        serializer_registry=serializer_registry
    )
    assert runtime.serializer_registry is serializer_registry


def test_runtime_boundary_hooks_use_owned_registry() -> None:
    schema_id = SchemaId("agentlane.workflow.task.model.v1")
    serializer_registry = SerializerRegistry()
    serializer_registry.register(
        PydanticJsonSerializer(
            schema_id=schema_id,
            model_type=TaskModel,
        )
    )
    runtime: RuntimeEngine = SingleThreadedRuntimeEngine(
        serializer_registry=serializer_registry
    )
    payload = Payload(
        schema_name=schema_id.value,
        content_type="application/json",
        format=PayloadFormat.JSON,
        data=TaskModel(name="compile"),
    )

    wire_payload = runtime.payload_to_wire_payload(payload)
    restored_payload = runtime.wire_payload_to_payload(wire_payload)

    assert restored_payload.data == TaskModel(name="compile")
    assert restored_payload.format == PayloadFormat.JSON


def test_runtime_boundary_hooks_fail_for_unknown_serializer() -> None:
    runtime: RuntimeEngine = SingleThreadedRuntimeEngine(
        serializer_registry=SerializerRegistry(auto_register_defaults=False)
    )
    payload = Payload(
        schema_name="agentlane.workflow.task.model.v1",
        content_type="application/json",
        format=PayloadFormat.JSON,
        data=TaskModel(name="compile"),
    )

    with pytest.raises(UnknownSerializerError):
        _ = runtime.payload_to_wire_payload(payload)


def test_registry_is_thread_safe_for_concurrent_register_and_lookup() -> None:
    registry = SerializerRegistry()
    schema_id = SchemaId("agentlane.workflow.task.model.v1")
    serializer = PydanticJsonSerializer(
        schema_id=schema_id,
        model_type=TaskModel,
    )
    errors: list[Exception] = []

    def register_worker() -> None:
        try:
            for _ in range(300):
                registry.register(serializer, replace=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def lookup_worker() -> None:
        try:
            for _ in range(300):
                has_serializer = registry.has(
                    schema_id=schema_id,
                    content_type="application/json",
                )
                if not has_serializer:
                    errors.append(RuntimeError("Serializer unexpectedly missing."))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    registry.register(serializer)
    threads = [
        Thread(target=register_worker),
        Thread(target=lookup_worker),
        Thread(target=lookup_worker),
        Thread(target=lookup_worker),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
