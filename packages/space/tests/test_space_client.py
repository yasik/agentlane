"""Exercise the native SPACE HTTP contract without a live deployment."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentlane_space import SpaceError, SpaceErrorKind, SpaceSandbox

from agentlane.runtime import CancellationToken

_FIXTURES = Path(__file__).parent / "fixtures"


def _frame(**fields: object) -> bytes:
    return b"data: " + json.dumps(fields).encode() + b"\n\n"


def _exit(*, status: str = "exited", code: int = 0) -> bytes:
    return _frame(
        type="process_exited", process_id="proc_example", exit_code=code, status=status
    )


def _output(data: bytes, *, kind: str = "stdout", seq: int = 1) -> bytes:
    return _frame(type=kind, seq=seq, data_base64=base64.b64encode(data).decode())


class EventBytes(httpx.AsyncByteStream):
    """Deliver transport chunks and optionally hold an open event stream."""

    def __init__(
        self, chunks: Sequence[bytes], *, release: asyncio.Event | None = None
    ) -> None:
        self.chunks = chunks
        self.release = release
        self.entered = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class NativeAPI:
    """A small native API fixture with separate command and attachment origins."""

    def __init__(self, stream: EventBytes | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.created: dict[str, Any] = json.loads(
            (_FIXTURES / "command_response.json").read_text()
        )
        self.stream = stream or EventBytes([(_FIXTURES / "events.sse").read_bytes()])
        self.creation_status = 201
        self.creation_error: Exception | None = None
        self.stream_status = 200
        self.content_type = "text/event-stream; charset=utf-8"
        self.kill_entered = asyncio.Event()
        self.kill_release: asyncio.Event | None = None
        self.transport = httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/commands"):
            if self.creation_error is not None:
                raise self.creation_error
            return httpx.Response(
                self.creation_status,
                json=self.created,
                headers={"Location": "https://unexpected.example/redirect"},
            )
        if request.url.path.endswith(":kill"):
            self.kill_entered.set()
            if self.kill_release is not None:
                await self.kill_release.wait()
            return httpx.Response(200, json={"ok": True})
        if request.url.path.endswith("/events"):
            return httpx.Response(
                self.stream_status,
                headers={
                    "Content-Type": self.content_type,
                    "Location": "https://unexpected.example/redirect",
                },
                stream=self.stream,
            )
        pytest.fail(f"Unexpected SPACE request: {request.method} {request.url}")

    def sandbox(self, *, cleanup_timeout: float = 0.2) -> SpaceSandbox:
        return SpaceSandbox(
            api_url="https://api.example",
            data_plane_url="https://events.example",
            token="fixture-management-token",  # noqa: S106 - Test-only credential.
            sandbox_id="sbx_example",
            transport=self.transport,
            cleanup_timeout=cleanup_timeout,
        )


async def _execute(
    sandbox: SpaceSandbox,
    *,
    token: CancellationToken | None = None,
    output_limit: int = 1024 * 1024,
    tail: bool = False,
) -> Any:
    return await sandbox.execute(
        ("/bin/bash", "-lc", "printf hello"),
        cwd="/guest/workspace",
        cancellation_token=token or CancellationToken(),
        output_limit=output_limit,
        tail=tail,
    )


@pytest.mark.asyncio
async def test_native_fixture_preserves_tokens_bytes_identity_and_close() -> None:
    api = NativeAPI()
    async with api.sandbox() as sandbox:
        result = await _execute(sandbox)
        identity = sandbox.last_command
        assert identity is not None
        assert identity.process_id == "proc_example"
        assert identity.session_id == "proc_example"
        assert not sandbox.needs_recovery

    assert result.stdout == b"Hello SPACE\n"
    assert result.stderr == b"warning\n"
    assert result.combined == b"Hello warning\nSPACE\n"
    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.truncated
    assert api.stream.closed
    assert len(api.requests) == 2
    creation, attachment = api.requests
    assert creation.method == "POST"
    assert str(creation.url) == "https://api.example/v1/sandboxes/sbx_example/commands"
    assert creation.headers["Authorization"] == "Bearer fixture-management-token"
    assert creation.headers["Idempotency-Key"] == identity.idempotency_key
    assert json.loads(creation.content) == {
        "command": ["/bin/bash", "-lc", "printf hello"],
        "cwd": "/guest/workspace",
        "timeout": "60.000000000s",
        "env": {},
        "stdin": False,
    }
    assert attachment.method == "GET"
    assert attachment.url.host == "events.example"
    assert attachment.headers["Authorization"] == "Bearer fixture-attachment-token"
    assert "Idempotency-Key" not in attachment.headers
    with pytest.raises(SpaceError) as error:
        await _execute(sandbox)
    assert error.value.kind is SpaceErrorKind.CLOSED
    assert len(api.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute_attach", [False, True])
async def test_configured_api_prefix_and_user_reach_native_endpoints(
    absolute_attach: bool,
) -> None:
    api = NativeAPI()
    path = "/stream/v1/sessions/proc_example/events"
    api.created["attach"]["events_url"] = (
        "https://events.example" + path if absolute_attach else path
    )
    async with SpaceSandbox(
        api_url="https://api.example/api/v1/",
        data_plane_url="https://events.example/stream/v1/",
        token=lambda: "rotating-management-token",
        sandbox_id="sbx_example",
        user="sandbox-user",
        transport=api.transport,
    ) as sandbox:
        await _execute(sandbox)
    assert api.requests[0].url.path == "/api/v1/sandboxes/sbx_example/commands"
    assert json.loads(api.requests[0].content)["user"] == "sandbox-user"
    assert api.requests[1].url.path == path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "destination",
    [
        "https://unexpected.example/v1/sessions/proc_example/events",
        "/v1/sessions/proc_other/events",
        "https://events.example/v1/sessions/proc_example/events?token=unexpected",
    ],
)
async def test_untrusted_attachment_destination_is_rejected(destination: str) -> None:
    api = NativeAPI()
    api.created["attach"]["events_url"] = destination
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.PROTOCOL
        assert sandbox.needs_recovery
    assert len(api.requests) == 2
    assert all(request.url.host == "api.example" for request in api.requests)
    assert api.requests[1].url.path.endswith(":kill")


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_step", ["create", "attach"])
async def test_redirect_is_not_followed_with_credentials(redirect_step: str) -> None:
    api = NativeAPI()
    if redirect_step == "create":
        api.creation_status = 307
    else:
        api.stream_status = 307
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError):
            await _execute(sandbox)
        assert sandbox.needs_recovery
    assert all(request.url.host != "unexpected.example" for request in api.requests)
    assert sum(request.url.path.endswith("/commands") for request in api.requests) == 1


@pytest.mark.asyncio
async def test_sse_byte_chunks_multiline_data_and_unknown_events() -> None:
    events = (
        b": heartbeat\r\n\r\n"
        b"event: stdout\r\nid: 1\r\ndata: {\r\n"
        b'data: "type":"stdout","seq":1,\r\n'
        b'data: "data_base64":"AP/DqQ=="}\r\n\r\n'
        + _frame(type="future_event", extra="ignored")
        + _exit()
    )
    api = NativeAPI(EventBytes([events[i : i + 1] for i in range(len(events))]))
    async with api.sandbox() as sandbox:
        result = await _execute(sandbox)
    assert result.stdout == b"\x00\xff\xc3\xa9"
    assert result.combined == result.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "resync_required"])
async def test_stream_error_remains_failure_when_followed_by_exit(kind: str) -> None:
    api = NativeAPI(EventBytes([_output(b"partial"), _frame(type=kind), _exit()]))
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.PROTOCOL
        assert sandbox.needs_recovery
        request_count = len(api.requests)
        with pytest.raises(SpaceError) as recovery:
            await _execute(sandbox)
        assert recovery.value.kind is SpaceErrorKind.RECOVERY_REQUIRED
        assert len(api.requests) == request_count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        _output(b"no exit"),
        b'data: {"type":"process_exited"}',
        b"data: invalid json\n\n",
        _frame(type="stdout", data_base64="invalid%%%") + _exit(),
        _frame(type="stdout") + _exit(),
        _exit() + _output(b"after exit"),
        _exit() + _exit(),
        _frame(
            type="process_exited", process_id="proc_other", exit_code=0, status="exited"
        ),
        _frame(
            type="process_exited",
            process_id="proc_example",
            exit_code=True,
            status="exited",
        ),
        _exit(status="unknown"),
    ],
)
async def test_incomplete_or_invalid_stream_requires_recovery(events: bytes) -> None:
    api = NativeAPI(EventBytes([events]))
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.PROTOCOL
        assert sandbox.needs_recovery
    assert api.stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("with_newline", [False, True])
async def test_oversized_sse_line_is_rejected(with_newline: bool) -> None:
    data = b"data: " + b"x" * (4 * 1024 * 1024)
    api = NativeAPI(EventBytes([data + (b"\n\n" if with_newline else b"")]))
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.PROTOCOL


@pytest.mark.asyncio
async def test_complete_result_limit_applies_to_combined_output() -> None:
    api = NativeAPI(
        EventBytes([_output(b"1234"), _output(b"5678", kind="stderr", seq=2), _exit()])
    )
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox, output_limit=6)
        assert error.value.kind is SpaceErrorKind.PROTOCOL


@pytest.mark.asyncio
async def test_tail_capture_preserves_binary_bytes_and_stream_order() -> None:
    api = NativeAPI(
        EventBytes(
            [
                _output(b"12345"),
                _output(b"\x00\xff", kind="stderr", seq=2),
                _output(b"89", seq=3),
                _exit(),
            ]
        )
    )
    async with api.sandbox() as sandbox:
        result = await _execute(sandbox, output_limit=5, tail=True)
    assert result.stdout == b"34589"
    assert result.stderr == b"\x00\xff"
    assert result.combined == b"5\x00\xff89"
    assert result.truncated


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["exited", "timed_out"])
async def test_timeout_uses_exit_status_instead_of_exit_code(status: str) -> None:
    api = NativeAPI(EventBytes([_output(b"retained"), _exit(status=status, code=124)]))
    async with api.sandbox() as sandbox:
        result = await _execute(sandbox)
    assert result.exit_code == 124
    assert result.timed_out is (status == "timed_out")
    assert result.stdout == b"retained"


@pytest.mark.asyncio
async def test_cancellation_before_dispatch_sends_no_request() -> None:
    api = NativeAPI()
    token = CancellationToken()
    token.cancel()
    async with api.sandbox() as sandbox:
        with pytest.raises(asyncio.CancelledError):
            await _execute(sandbox, token=token)
        assert not sandbox.needs_recovery
        assert sandbox.last_command is None
    assert api.requests == []


@pytest.mark.asyncio
async def test_cancellation_while_queued_does_not_start_another_command() -> None:
    release = asyncio.Event()
    api = NativeAPI(EventBytes([_exit()], release=release))
    async with api.sandbox() as sandbox:
        active = asyncio.create_task(_execute(sandbox))
        await asyncio.wait_for(api.stream.entered.wait(), 1)
        token = CancellationToken()
        queued = asyncio.create_task(_execute(sandbox, token=token))
        await asyncio.sleep(0)
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, 1)
        assert len(api.requests) == 2
        assert not sandbox.needs_recovery
        release.set()
        assert (await asyncio.wait_for(active, 1)).exit_code == 0


@pytest.mark.asyncio
async def test_active_token_cancellation_closes_stream_and_requests_leader_kill() -> (
    None
):
    api = NativeAPI(EventBytes([], release=asyncio.Event()))
    token = CancellationToken()
    async with api.sandbox() as sandbox:
        active = asyncio.create_task(_execute(sandbox, token=token))
        await asyncio.wait_for(api.stream.entered.wait(), 1)
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 1)
        assert sandbox.needs_recovery
    assert api.stream.closed
    assert len(api.requests) == 3
    kill = api.requests[-1]
    assert kill.url.path == "/v1/sandboxes/sbx_example/processes/proc_example:kill"
    assert kill.headers["Authorization"] == "Bearer fixture-management-token"
    assert "Idempotency-Key" not in kill.headers


@pytest.mark.asyncio
async def test_repeated_task_cancellation_waits_for_bounded_cleanup_before_unlock() -> (
    None
):
    api = NativeAPI(EventBytes([], release=asyncio.Event()))
    api.kill_release = asyncio.Event()
    async with api.sandbox(cleanup_timeout=0.05) as sandbox:
        active = asyncio.create_task(_execute(sandbox))
        await asyncio.wait_for(api.stream.entered.wait(), 1)
        active.cancel()
        await asyncio.wait_for(api.kill_entered.wait(), 1)
        active.cancel()
        closing = asyncio.create_task(sandbox.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        queued = asyncio.create_task(_execute(sandbox))
        with pytest.raises(SpaceError) as error:
            await asyncio.wait_for(queued, 1)
        assert error.value.kind is SpaceErrorKind.RECOVERY_REQUIRED
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 1)
        await asyncio.wait_for(closing, 1)
        assert sandbox.needs_recovery
    assert len(api.requests) == 3
    assert api.stream.closed


@pytest.mark.asyncio
async def test_lost_creation_response_retains_original_key_without_retry() -> None:
    api = NativeAPI()
    api.creation_error = httpx.ReadTimeout("private server detail")
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.TRANSPORT
        assert "private server detail" not in str(error.value)
        assert sandbox.needs_recovery
        identity = sandbox.last_command
        assert identity is not None
        assert identity.process_id is None
        assert identity.idempotency_key == api.requests[0].headers["Idempotency-Key"]
        with pytest.raises(SpaceError) as recovery:
            await _execute(sandbox)
        assert recovery.value.kind is SpaceErrorKind.RECOVERY_REQUIRED
    assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["attach", "process_id", "session_id"])
async def test_partial_creation_response_retains_valid_recovery_identifiers(
    invalid_field: str,
) -> None:
    api = NativeAPI()
    api.created[invalid_field] = None
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError):
            await _execute(sandbox)
        identity = sandbox.last_command
        assert identity is not None
        assert identity.session_id == (
            None if invalid_field == "session_id" else "proc_example"
        )
        assert identity.process_id == (
            None if invalid_field == "process_id" else "proc_example"
        )
        assert sandbox.needs_recovery
    assert not any(request.method == "GET" for request in api.requests)


@pytest.mark.asyncio
async def test_queued_command_retains_validated_environment_snapshot() -> None:
    release = asyncio.Event()
    api = NativeAPI(EventBytes([_exit()], release=release))
    environment = {"EXAMPLE": "original"}
    async with api.sandbox() as sandbox:
        active = asyncio.create_task(_execute(sandbox))
        await asyncio.wait_for(api.stream.entered.wait(), 1)
        queued = asyncio.create_task(
            sandbox.execute(
                ("true",),
                cwd="/guest/workspace",
                cancellation_token=CancellationToken(),
                env=environment,
            )
        )
        await asyncio.sleep(0)
        environment["EXAMPLE"] = "changed"
        release.set()
        await asyncio.wait_for(active, 1)
        await asyncio.wait_for(queued, 1)
    assert len(api.requests) == 4
    assert json.loads(api.requests[2].content)["env"] == {"EXAMPLE": "original"}


@pytest.mark.asyncio
async def test_rejected_creation_is_not_retried_and_does_not_poison_handle() -> None:
    api = NativeAPI()
    api.creation_status = 429
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError):
            await _execute(sandbox)
        assert not sandbox.needs_recovery
        assert len(api.requests) == 1
        first_identity = sandbox.last_command
        api.creation_status = 201
        await _execute(sandbox)
        assert sandbox.last_command != first_identity
    assert len(api.requests) == 3


@pytest.mark.asyncio
async def test_command_limit_includes_argument_terminators_before_dispatch() -> None:
    api = NativeAPI()
    arguments = ("x" * (90 * 1024), "x" * (90 * 1024), "x" * (76 * 1024))
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await sandbox.execute(
                arguments,
                cwd="/guest/workspace",
                cancellation_token=CancellationToken(),
            )
        assert error.value.kind is SpaceErrorKind.INPUT_LIMIT
        assert not sandbox.needs_recovery
    assert api.requests == []


@pytest.mark.asyncio
async def test_environment_key_limit_is_checked_before_dispatch() -> None:
    api = NativeAPI()
    async with api.sandbox() as sandbox:
        with pytest.raises(ValueError):
            await sandbox.execute(
                ("true",),
                cwd="/guest/workspace",
                cancellation_token=CancellationToken(),
                env={"K" * 129: "value"},
            )
        assert not sandbox.needs_recovery
    assert api.requests == []


@pytest.mark.asyncio
async def test_transport_deadline_requires_recovery_and_bounds_cleanup() -> None:
    api = NativeAPI(EventBytes([], release=asyncio.Event()))
    api.kill_release = asyncio.Event()
    async with SpaceSandbox(
        api_url="https://api.example",
        data_plane_url="https://events.example",
        token=lambda: "fixture-management-token",
        sandbox_id="sbx_example",
        default_timeout=0.01,
        request_timeout=0.01,
        cleanup_timeout=0.01,
        transport=api.transport,
    ) as sandbox:
        with pytest.raises(SpaceError) as error:
            await asyncio.wait_for(_execute(sandbox), 1)
        assert error.value.kind is SpaceErrorKind.TRANSPORT
        assert sandbox.needs_recovery
        assert api.kill_entered.is_set()
    assert len(api.requests) == 3
    assert api.stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [None, 1, 3, True])
async def test_missing_duplicate_or_discontinuous_output_requires_recovery(
    sequence: object,
) -> None:
    fields: dict[str, object] = {"type": "stderr", "data_base64": "Yg=="}
    if sequence is not None:
        fields["seq"] = sequence
    api = NativeAPI(EventBytes([_output(b"a"), _frame(**fields), _exit()]))
    async with api.sandbox() as sandbox:
        with pytest.raises(SpaceError) as error:
            await _execute(sandbox)
        assert error.value.kind is SpaceErrorKind.PROTOCOL
        assert sandbox.needs_recovery
