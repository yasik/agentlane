"""Run guest commands behind the public native SPACE transport boundary."""

import asyncio
import base64
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentlane_space import SpaceSandbox
from pydantic import BaseModel, Field


class _Command(BaseModel):
    command: list[str]
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)


class LocalSpace:
    """A fake SPACE service that actually runs the requested guest commands."""

    def __init__(self) -> None:
        self.commands: dict[str, _Command] = {}
        self.environment: dict[str, str] = {}
        self.transform_response: Callable[[bytes], bytes] | None = None
        self.fail_stream = False
        self.fail_connect = False
        self.stream_gate: asyncio.Event | None = None
        self.stream_entered = asyncio.Event()
        self.worker_response: bytes | None = None
        self.worker_exit_code: int | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/commands"):
            command = _Command.model_validate_json(request.content)
            identifier = f"proc_{len(self.commands) + 1}"
            self.commands[identifier] = command
            return httpx.Response(
                201,
                json={
                    "sandbox_id": "sbx_test",
                    "process_id": identifier,
                    "session_id": identifier,
                    "attach": {
                        "events_url": f"/v1/sessions/{identifier}/events",
                        "token": "stream-secret",
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith(":kill"):
            return httpx.Response(200, json={"ok": True})
        if request.method == "GET":
            identifier = request.url.path.split("/")[-2]
            if self.fail_connect:
                raise httpx.ConnectError(
                    "Test stream connection failed", request=request
                )
            if self.fail_stream:
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=b""
                )
            if self.stream_gate is not None:
                self.stream_entered.set()
                await self.stream_gate.wait()
            command = self.commands[identifier]
            argv = command.command.copy()
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=command.cwd,
                env={**os.environ, **self.environment, **command.env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=15
                )
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            exit_code = process.returncode
            if "agentlane_space._worker" in argv:
                if self.transform_response is not None:
                    stdout = self.transform_response(stdout)
                if self.worker_response is not None:
                    stdout = self.worker_response
                if self.worker_exit_code is not None:
                    exit_code = self.worker_exit_code
            frames: list[dict[str, Any]] = []
            for stream, data in (("stdout", stdout), ("stderr", stderr)):
                # Split protocol output across events, including UTF-8 bytes.
                for offset in range(0, len(data), 151):
                    frames.append(
                        {
                            "type": stream,
                            "seq": len(frames) + 1,
                            "data_base64": base64.b64encode(
                                data[offset : offset + 151]
                            ).decode(),
                        }
                    )
            frames.append(
                {
                    "type": "process_exited",
                    "process_id": identifier,
                    "exit_code": exit_code,
                    "status": "exited",
                }
            )
            payload = b"".join(
                b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")


def sandbox_for(service: LocalSpace) -> SpaceSandbox:
    return SpaceSandbox(
        api_url="https://api.example.com",
        data_plane_url="https://events.example.com",
        token="management-secret",  # noqa: S106 - Fixed mock transport credential.
        sandbox_id="sbx_test",
        python_executable=sys.executable,
        transport=httpx.MockTransport(service.handle),
    )


@pytest.fixture
def local_space() -> Iterator[LocalSpace]:
    service = LocalSpace()
    yield service
    # These paths came from commands executed by this fixture, not other tests.
    for command in service.commands.values():
        for argument in command.command:
            if argument.startswith(
                "/tmp/agentlane-bash-"  # noqa: S108 - Test-owned guest log.
            ) and argument.endswith(".log"):
                Path(argument).unlink(missing_ok=True)
