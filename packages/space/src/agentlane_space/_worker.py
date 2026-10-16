"""Run one native file tool in a prepared SPACE guest environment."""

import asyncio
import shutil
import sys
import tempfile
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError

from agentlane.harness.tools import base_harness_tools
from agentlane.models import Tool, ToolExecutionContext, ToolFailure
from agentlane.runtime import CancellationToken

from ._protocol import (
    FILE_TOOL_NAMES,
    RESULT_LIMIT,
    ToolMetadata,
    WorkerError,
    WorkerInfo,
    WorkerRequest,
    WorkerResponse,
    build_identity,
    decode_request,
)


def _failure(message: str, *, kind: str) -> WorkerResponse:
    return WorkerResponse(text=message, error=WorkerError(message=message, kind=kind))


def _check_environment() -> WorkerResponse:
    """Check the prepared template and report guest-owned tool metadata."""
    if sys.version_info[:2] != (3, 12):
        return _failure("SPACE worker requires Python 3.12", kind="preflight")

    for executable in ("/bin/bash", "rg", "tee"):
        if shutil.which(executable) is None:
            return _failure(
                f"SPACE worker requires executable: {executable}", kind="preflight"
            )

    # Probe writes as the actual guest user. Permission bits alone do not
    # establish access through mounts or ACLs. Temporary files clean up the probe.
    cwd = Path.cwd()
    try:
        with tempfile.TemporaryFile(dir=cwd):
            pass
    except OSError:
        return _failure("SPACE worker requires a writable cwd", kind="preflight")

    try:
        with tempfile.TemporaryFile(dir="/tmp"):  # noqa: S108 - Secure temporary file.
            pass
    except OSError:
        return _failure(
            "SPACE worker requires writable output storage", kind="preflight"
        )

    version, fingerprint = build_identity()

    # Path-dependent descriptions must come from the guest's filesystem and
    # home directory, not from the host that constructs the tool bundle.
    definitions = base_harness_tools(cwd=cwd, include=FILE_TOOL_NAMES)
    metadata = [
        ToolMetadata(
            name=definition.tool.name,
            description=definition.tool.description,
            prompt_snippet=definition.prompt_snippet,
            prompt_guidelines=list(definition.prompt_guidelines),
            parameters_schema=definition.tool.schema["parameters"],
        )
        for definition in definitions
    ]
    return WorkerResponse(
        info=WorkerInfo(
            version=version,
            fingerprint=fingerprint,
            cwd=str(cwd),
            tools=metadata,
        )
    )


async def execute_request(request: WorkerRequest) -> WorkerResponse:
    """Execute only the selected native file handler with guest path rules.

    Host callbacks, credentials, and live run state are not part of this protocol.
    Expected file-tool error strings remain unchanged; only native `ToolFailure`
    results carry typed tool failures.
    """
    try:
        if request.tool == "check":
            if request.arguments:
                return _failure(
                    "SPACE check does not accept arguments", kind="protocol"
                )

            return _check_environment()

        # Recreate the core tool inside the guest so its file operations use
        # the guest filesystem. The host proxy only transports this request.
        definition = base_harness_tools(cwd=Path.cwd(), include=(request.tool,))[0]
        tool = cast(Tool[BaseModel, str], definition.tool)
        if not isinstance(definition.tool, Tool):
            return _failure("SPACE worker tool is not executable", kind="worker")

        # Validate again in the guest before any file handler runs. The worker
        # accepts serialized data, not an already validated host Python object.
        arguments = tool.args_type().model_validate(request.arguments)
        context = ToolExecutionContext(
            run_id=request.context.run_id,
            agent_name=request.context.agent_name,
            tool_call_id=request.context.tool_call_id,
        )
        result = await tool.run(arguments, CancellationToken(), context=context)
        text = tool.return_value_as_string(result)
        error = (
            WorkerError(message=result.error.message, kind=result.error.kind)
            if isinstance(result, ToolFailure)
            else None
        )
        return WorkerResponse(text=text, error=error)
    except ValidationError:
        return _failure("invalid SPACE worker tool arguments", kind="protocol")
    except Exception:
        return _failure("SPACE worker failed to execute the request", kind="worker")


def main(argv: Sequence[str] | None = None) -> int:
    """Read one argv request and write exactly one bounded JSON response."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    exit_code = 0
    try:
        if len(arguments) != 1:
            raise ValueError("expected one encoded worker request")

        request = decode_request(arguments[0])
        # Library diagnostics must not become part of the JSON output stream.
        with redirect_stdout(sys.stderr):
            response = asyncio.run(execute_request(request))
    except ValueError:
        response = _failure("invalid SPACE worker request", kind="protocol")
        exit_code = 1
    except Exception:
        response = _failure("SPACE worker failed to execute the request", kind="worker")
        exit_code = 1

    # Worker/protocol faults invalidate the result. Native tool failures remain
    # completed calls with typed error data rather than worker process failures.
    if response.error is not None and response.error.kind in {
        "protocol",
        "preflight",
        "worker",
    }:
        exit_code = 1

    payload = response.model_dump_json().encode("utf-8")

    # JSON escaping can expand tool text. Count serialized bytes and the final
    # newline; a partial envelope is never a valid result for the host.
    if len(payload) + 1 > RESULT_LIMIT:
        response = _failure(
            "SPACE worker result exceeds the size limit", kind="protocol"
        )
        payload = response.model_dump_json().encode("utf-8")
        exit_code = 1

    if exit_code:
        print("SPACE worker request failed", file=sys.stderr)

    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
