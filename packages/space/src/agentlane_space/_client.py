"""Attach-only native SPACE command client."""

import asyncio
import json
import math
import posixpath
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from typing import Any, Self, cast
from uuid import uuid4

import httpx
from pydantic import ValidationError

from agentlane.runtime import CancellationToken

from ._errors import SpaceError, SpaceErrorKind
from ._protocol import (
    RESULT_LIMIT,
    WorkerInfo,
    WorkerRequest,
    build_identity,
    decode_response,
    encode_request,
)
from ._stream import OutputCollector, event_frames
from ._types import CommandIdentity, CommandResult, CreatedCommand, ServerFrame


class SpaceSandbox:
    """Connect tools to an existing SPACE sandbox.

    The application owns the sandbox lifecycle. Closing this handle closes its
    HTTP client only. An uncertain command outcome blocks further execution on
    the handle; the application must recover the sandbox before creating a new
    handle. Token suppliers are called for each management request.

    Args:
        api_url: Management origin or API prefix, such as `https://api.example.com/v1`.
        data_plane_url: Event origin or prefix, such as `https://events.example.com/v1`.
        token: Management bearer token or synchronous token supplier.
        sandbox_id: Identifier of an existing, running sandbox.
        default_timeout: Default server command deadline in seconds.
        request_timeout: Management request deadline in seconds.
        cleanup_timeout: Maximum time for a termination request in seconds.
        user: Optional guest user, shared by checks and tools.
        python_executable: Guest Python executable containing the matching extension.
        transport: Optional HTTP transport for tests or application networking.
    """

    def __init__(
        self,
        *,
        api_url: str,
        data_plane_url: str,
        token: str | Callable[[], str],
        sandbox_id: str,
        default_timeout: float = 60.0,
        request_timeout: float = 30.0,
        cleanup_timeout: float = 5.0,
        user: str | None = None,
        python_executable: str = "python3",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if re.fullmatch(r"sbx_[A-Za-z0-9_-]+", sandbox_id) is None:
            raise ValueError("sandbox_id must be a SPACE sandbox identifier.")
        for value in (default_timeout, request_timeout, cleanup_timeout):
            _positive_timeout(value)
        if not python_executable or "\0" in python_executable:
            raise ValueError("python_executable must name a guest executable.")
        if user is not None and (not user or "\0" in user):
            raise ValueError("user must name a guest user.")

        self._api_url = _api_prefix(api_url)
        self._data_plane_url = _api_prefix(data_plane_url)
        self._token = token
        self.sandbox_id = sandbox_id
        self.default_timeout = default_timeout
        self.python_executable = python_executable
        self._request_timeout = request_timeout
        self._cleanup_timeout = cleanup_timeout
        self._user = user
        self._http = httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False
        )
        self._lock = asyncio.Lock()
        self._checked: dict[str, WorkerInfo] = {}
        self._needs_recovery = False
        self._closed = False
        self._last_command: CommandIdentity | None = None

    @property
    def needs_recovery(self) -> bool:
        """Whether an uncertain outcome prevents further use of this handle."""
        return self._needs_recovery

    @property
    def last_command(self) -> CommandIdentity | None:
        """Return the last command's recovery identifiers, without credentials."""
        return self._last_command

    def mark_unusable(self) -> None:
        """Block this handle when the application or worker cannot confirm an outcome."""
        self._needs_recovery = True

    async def __aenter__(self) -> Self:
        self._ensure_available()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close HTTP resources without deleting or changing the sandbox."""
        async with self._lock:
            self._closed = True
            self._checked.clear()
            await self._http.aclose()

    async def check(self, *, cwd: str) -> None:
        """Verify the guest build, tools, and working directory before use."""
        directory = guest_cwd(cwd)

        # Both callbacks run under the command lock. Clearing readiness earlier
        # would let an older check restore it after a newer check has failed.
        def invalidate() -> None:
            self._checked.pop(directory, None)

        def validate(result: CommandResult) -> None:
            try:
                response = decode_response(result.stdout)
                info = response.info
                version, fingerprint = build_identity()
                if (
                    result.exit_code != 0
                    or result.timed_out
                    or response.error is not None
                    or info is None
                    or info.version != version
                    or info.fingerprint != fingerprint
                ):
                    raise ValueError("Guest check failed.")
            except (ValueError, ValidationError):
                raise SpaceError(
                    "SPACE guest check failed. Install the matching AgentLane build and guest tools.",
                    kind=SpaceErrorKind.PROTOCOL,
                ) from None
            self._checked[directory] = info

        request = encode_request(WorkerRequest(tool="check"))
        await self._execute(
            self.worker_command(request),
            cwd=directory,
            cancellation_token=CancellationToken(),
            before_execute=invalidate,
            validate_result=validate,
        )

    def require_checked(self, cwd: str) -> None:
        """Reject tool use before a matching guest check succeeds."""
        self.checked_info(cwd)

    def checked_info(self, cwd: str) -> WorkerInfo:
        """Return guest metadata for a checked working directory."""
        self._ensure_available()
        info = self._checked.get(guest_cwd(cwd))
        if info is None:
            raise SpaceError(
                "Call await sandbox.check(cwd=...) before constructing or running SPACE tools.",
                kind=SpaceErrorKind.NOT_CHECKED,
            )
        return info

    def worker_command(self, encoded_request: str) -> tuple[str, ...]:
        """Build a literal argv request for the installed guest worker."""
        # Isolated Python ignores cwd modules and PYTHONPATH. Project files in
        # the guest must not replace the installed worker during tool startup.
        return (
            self.python_executable,
            "-I",
            "-m",
            "agentlane_space._worker",
            encoded_request,
        )

    async def execute(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        cancellation_token: CancellationToken,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        output_limit: int = RESULT_LIMIT,
        tail: bool = False,
        require_checked: bool = False,
    ) -> CommandResult:
        """Execute one bounded command with serialized access and no retries.

        Environment values override the guest environment; they do not replace
        it. Set require_checked to revalidate readiness at serialized dispatch.
        """
        return await self._execute(
            command,
            cwd=cwd,
            cancellation_token=cancellation_token,
            timeout=timeout,
            env=env,
            output_limit=output_limit,
            tail=tail,
            before_execute=(
                (lambda: self.require_checked(cwd)) if require_checked else None
            ),
        )

    async def execute_worker(
        self,
        encoded_request: str,
        *,
        cwd: str,
        cancellation_token: CancellationToken,
    ) -> CommandResult:
        """Run a file worker and validate its result before another call starts."""
        return await self._execute(
            self.worker_command(encoded_request),
            cwd=cwd,
            cancellation_token=cancellation_token,
            before_execute=lambda: self.require_checked(cwd),
            validate_result=self._validate_worker_result,
        )

    async def _execute(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        cancellation_token: CancellationToken,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        output_limit: int = RESULT_LIMIT,
        tail: bool = False,
        before_execute: Callable[[], None] | None = None,
        validate_result: Callable[[CommandResult], None] | None = None,
    ) -> CommandResult:
        effective_timeout = self.default_timeout if timeout is None else timeout
        _positive_timeout(effective_timeout)
        directory = guest_cwd(cwd)

        # Callers can mutate their inputs while this command waits. Dispatch
        # the same argument and environment values that passed validation.
        if isinstance(command, str):
            raise ValueError("command must be an argument sequence.")
        arguments = tuple(command)
        environment = dict(env) if env is not None else None
        _validate_command(arguments, environment)
        if output_limit <= 0:
            raise ValueError("output_limit must be positive.")
        self._ensure_available()
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        operation = asyncio.create_task(
            self._execute_serialized(
                arguments,
                cwd=directory,
                cancellation_token=cancellation_token,
                timeout=effective_timeout,
                env=environment,
                output_limit=output_limit,
                tail=tail,
                before_execute=before_execute,
                validate_result=validate_result,
            )
        )
        cancelled = asyncio.create_task(cancellation_token.wait_cancelled())
        try:
            done, _ = await asyncio.wait(
                {operation, cancelled}, return_when=asyncio.FIRST_COMPLETED
            )
            if operation in done:
                # Prefer a completed operation if cancellation becomes ready in
                # the same loop iteration. Preserve its result or original failure.
                return await operation
            operation.cancel()
            await _settle_cancelled(operation)
            raise asyncio.CancelledError
        except asyncio.CancelledError:
            # Caller task cancellation and token cancellation must both wait
            # for bounded remote cleanup before releasing command ownership.
            operation.cancel()
            await _settle_cancelled(operation)
            raise
        finally:
            cancelled.cancel()
            with suppress(asyncio.CancelledError):
                await cancelled

    async def _execute_serialized(
        self,
        command: tuple[str, ...],
        *,
        cwd: str,
        cancellation_token: CancellationToken,
        timeout: float,
        env: Mapping[str, str] | None,
        output_limit: int,
        tail: bool,
        before_execute: Callable[[], None] | None,
        validate_result: Callable[[CommandResult], None] | None,
    ) -> CommandResult:
        async with self._lock:
            self._ensure_available()
            if cancellation_token.is_cancelled:
                raise asyncio.CancelledError

            # A queued command can outlive a failed readiness check. Validate
            # again here, before assigning recovery IDs or sending a request.
            if before_execute is not None:
                before_execute()
            body: dict[str, Any] = {
                "command": list(command),
                "cwd": cwd,
                "timeout": f"{timeout:.9f}s",
                "env": dict(env or {}),
                "stdin": False,
            }
            if self._user is not None:
                body["user"] = self._user
            identity = CommandIdentity(idempotency_key=uuid4().hex)
            self._last_command = identity
            submitted = False
            try:
                headers = self._management_headers()
                headers["Idempotency-Key"] = identity.idempotency_key

                # Sending can succeed even when its response is lost. From
                # this point, failure requires recovery unless rejection is known.
                submitted = True
                async with asyncio.timeout(self._request_timeout):
                    response = await self._http.post(
                        self._sandbox_url + "/commands",
                        json=body,
                        headers=headers,
                        timeout=self._request_timeout,
                        follow_redirects=False,
                    )
                if 400 <= response.status_code < 500 and response.status_code != 408:
                    # A client rejection confirms no command was accepted.
                    # A request timeout does not give that confirmation.
                    submitted = False
                response.raise_for_status()
                payload = json.loads(response.content)

                # Retain valid identifiers even if attachment validation fails.
                if isinstance(payload, dict):
                    fields = cast(dict[str, object], payload)
                    self._last_command = replace(
                        identity,
                        process_id=_process_id(fields.get("process_id")),
                        session_id=_process_id(fields.get("session_id")),
                    )
                created = CreatedCommand.model_validate(payload)
                self._last_command = replace(
                    identity,
                    process_id=created.process_id,
                    session_id=created.session_id,
                )
                if created.sandbox_id != self.sandbox_id:
                    raise SpaceError(
                        "SPACE returned an unexpected sandbox identity.",
                        kind=SpaceErrorKind.PROTOCOL,
                    )
                events_url = self._events_url(created)
                async with asyncio.timeout(timeout + self._request_timeout):
                    result = await self._consume(
                        created, events_url, output_limit, tail
                    )
            except BaseException as exc:
                if submitted:
                    # Block queued commands before cleanup. A successful kill
                    # cannot reverse writes or prove that detached children exited.
                    self._needs_recovery = True
                    cleanup = asyncio.create_task(self._terminate_last_command())
                    await _settle_cleanup(cleanup)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if isinstance(exc, SpaceError):
                    raise
                if isinstance(exc, Exception):
                    raise SpaceError(
                        (
                            "SPACE command failed. Its outcome can be unknown; inspect the sandbox before reuse."
                            if submitted
                            else "SPACE rejected the command request."
                        ),
                        kind=SpaceErrorKind.TRANSPORT,
                    ) from None
                raise

            # Readiness and result validation share the command lock. A failed
            # completed check invalidates readiness without an uncertain command.
            if validate_result is not None:
                validate_result(result)
            return result

    def _validate_worker_result(self, result: CommandResult) -> None:
        try:
            _validate_worker_result(result)
        except SpaceError:
            # The process exited, but its file-operation result is untrusted.
            # Mark the handle before another queued command acquires the lock.
            self.mark_unusable()
            raise

    async def _consume(
        self, created: CreatedCommand, events_url: str, limit: int, tail: bool
    ) -> CommandResult:
        collector = OutputCollector(limit=limit, tail=tail)
        terminal: ServerFrame | None = None
        expected_sequence = 1

        # Only the scoped attachment token goes to the pinned event endpoint.
        # The management token stays on management requests.
        async with self._http.stream(
            "GET",
            events_url,
            headers={
                "Authorization": f"Bearer {created.attach.token}",
                "Accept": "text/event-stream",
            },
            timeout=None,
            follow_redirects=False,
        ) as response:
            response.raise_for_status()
            if (
                response.headers.get("content-type", "").split(";")[0].strip()
                != "text/event-stream"
            ):
                raise SpaceError(
                    "SPACE did not return an event stream.",
                    kind=SpaceErrorKind.PROTOCOL,
                )
            # Read through EOF rather than returning at exit. Later errors,
            # duplicate exits, or extra output invalidate the result.
            async for frame in event_frames(response):
                if frame.type in {"error", "resync_required"}:
                    raise SpaceError(
                        "SPACE reported incomplete command output.",
                        kind=SpaceErrorKind.PROTOCOL,
                    )
                if frame.type in {"stdout", "stderr"}:
                    # SPACE shares one sequence across both output streams.
                    # Reject gaps and duplicates before they can alter valid JSON.
                    if terminal is not None or frame.seq != expected_sequence:
                        raise SpaceError(
                            "SPACE returned out-of-order command output.",
                            kind=SpaceErrorKind.PROTOCOL,
                        )
                    expected_sequence += 1
                    collector.append(frame)
                elif frame.type == "process_exited":
                    if terminal is not None or frame.process_id != created.process_id:
                        raise SpaceError(
                            "SPACE returned an invalid command exit.",
                            kind=SpaceErrorKind.PROTOCOL,
                        )
                    terminal = frame

        if terminal is None:
            raise SpaceError(
                "SPACE stream ended without a confirmed exit.",
                kind=SpaceErrorKind.PROTOCOL,
            )
        return collector.result(terminal)

    def _events_url(self, created: CreatedCommand) -> str:
        expected = f"{self._data_plane_url}/sessions/{created.session_id}/events"
        path = httpx.URL(expected).raw_path.decode("ascii")

        # Accept SPACE's absolute or relative form, but never use a server-
        # supplied origin, query, or different session with the attachment token.
        if created.attach.events_url not in {expected, path}:
            raise SpaceError(
                "SPACE returned an unexpected attachment destination.",
                kind=SpaceErrorKind.PROTOCOL,
            )
        return expected

    @property
    def _sandbox_url(self) -> str:
        return f"{self._api_url}/sandboxes/{self.sandbox_id}"

    def _management_headers(self) -> dict[str, str]:
        token = self._token() if callable(self._token) else self._token
        if not token or not token.strip() or "\r" in token or "\n" in token:
            raise ValueError("A valid SPACE management token is required.")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    async def _terminate_last_command(self) -> None:
        identity = self._last_command
        if identity is None or identity.process_id is None:
            return
        try:
            async with asyncio.timeout(self._cleanup_timeout):
                await self._http.post(
                    f"{self._sandbox_url}/processes/{identity.process_id}:kill",
                    headers=self._management_headers(),
                    timeout=self._cleanup_timeout,
                    follow_redirects=False,
                )
        except Exception:
            # Recovery remains required even if this best-effort leader kill
            # succeeds. The application owns any stronger sandbox cleanup.
            return

    def _ensure_available(self) -> None:
        if self._closed:
            raise SpaceError("SPACE handle is closed.", kind=SpaceErrorKind.CLOSED)
        if self._needs_recovery:
            raise SpaceError(
                "SPACE handle requires application recovery before further execution.",
                kind=SpaceErrorKind.RECOVERY_REQUIRED,
            )


def _validate_worker_result(result: CommandResult) -> None:
    if result.timed_out:
        # A confirmed timeout can interrupt JSON output. The tool reports the
        # timeout directly; parsing partial JSON would hide that known outcome.
        return
    try:
        response = decode_response(result.stdout)
    except ValueError:
        raise SpaceError(
            "SPACE worker returned an invalid result; inspect the sandbox before reuse.",
            kind=SpaceErrorKind.PROTOCOL,
        ) from None
    if result.exit_code != 0 or response.info is not None:
        # Readiness metadata is valid only for check(), never as a tool result.
        raise SpaceError(
            "SPACE worker failed; inspect the sandbox before reuse.",
            kind=SpaceErrorKind.PROTOCOL,
        )


def guest_cwd(cwd: str) -> str:
    """Normalize a rooted guest directory without host filesystem access."""
    if not cwd.startswith("/") or cwd.startswith("//") or "\0" in cwd:
        raise ValueError("cwd must be an absolute POSIX guest path.")
    return posixpath.normpath(cwd)


def _api_prefix(value: str) -> str:
    url = httpx.URL(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.username
        or url.password
        or url.query
        or url.fragment
        or (url.scheme == "http" and url.host not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ValueError(
            "SPACE endpoints require HTTPS, or loopback HTTP, without credentials or query parameters."
        )
    normalized = str(url).rstrip("/")
    return normalized + "/v1" if url.path in {"", "/"} else normalized


def _positive_timeout(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Timeouts must be finite and greater than zero.")


def _process_id(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"proc_[A-Za-z0-9_-]+", value):
        return value
    return None


def _validate_command(command: Sequence[str], env: Mapping[str, str] | None) -> None:
    if isinstance(command, str) or not command or len(command) > 4096:
        raise ValueError("command must be a nonempty argument sequence.")
    sizes = [len(value.encode("utf-8")) for value in command]
    if any("\0" in value for value in command) or not command[0]:
        raise ValueError("Command arguments must not contain NUL bytes.")
    # SPACE counts argument terminators in its total argv budget. Enforce
    # both that budget and the per-argument limit before remote execution.
    if sum(size + 1 for size in sizes) > 256 * 1024 or any(
        size > 96 * 1024 for size in sizes
    ):
        raise SpaceError(
            "SPACE command exceeds the argument byte limit.",
            kind=SpaceErrorKind.INPUT_LIMIT,
        )
    if env is not None:
        if (
            len(env) > 256
            or any(
                not key
                or len(key.encode("utf-8")) > 128
                or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) is None
                or "=" in key
                or "\0" in key
                or "\0" in value
                or len(value.encode("utf-8")) > 8192
                for key, value in env.items()
            )
            or sum(len(k.encode()) + len(v.encode()) for k, v in env.items())
            > 64 * 1024
        ):
            raise ValueError(
                "Environment exceeds SPACE limits or contains invalid entries."
            )


async def _settle_cancelled(task: asyncio.Task[CommandResult]) -> None:
    # Repeated caller cancellations must not interrupt the operation's cleanup.
    # Observe its exception without replacing the caller's cancellation result.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()


async def _settle_cleanup(task: asyncio.Task[None]) -> None:
    # The kill request has its own deadline. Shield it until that deadline or
    # completion so a second cancellation cannot release the command lock early.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()
