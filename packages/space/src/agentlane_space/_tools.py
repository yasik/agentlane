"""Compose native harness tools for an attached SPACE sandbox."""

import asyncio
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel

from agentlane.harness.tools import (
    BashExecutionRequest,
    BashExecutionResult,
    HarnessToolDefinition,
    base_harness_tools,
    bash_tool,
)
from agentlane.models import Tool, ToolError, ToolExecutionContext, ToolFailure
from agentlane.runtime import CancellationToken

from ._bash import SpaceBashExecutor
from ._client import SpaceSandbox, guest_cwd
from ._errors import SpaceError, SpaceErrorKind
from ._protocol import (
    FILE_TOOL_NAMES,
    INPUT_LIMIT,
    WorkerContext,
    WorkerRequest,
    decode_response,
    encode_request,
)


def space_harness_tools(
    sandbox: SpaceSandbox,
    *,
    cwd: str,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> tuple[HarnessToolDefinition, ...]:
    """Build a SPACE-configured bundle from the core base-tool definitions.

    Calls `base_harness_tools()` with the supplied selectors, preserving tool
    names, argument schemas, and standard order. Replaces the five file-tool
    handlers with proxies to a guest worker, which runs the core file handlers.
    Bash uses the core Bash tool with a SPACE executor. `write_plan` and `agent`
    retain their host definitions.

    Pass the returned bundle to `HarnessToolsShim` in place of a local base-tool
    bundle. This factory does not modify tools that are already registered.
    The sandbox and guest user permissions are the boundary. Host permission
    callbacks are not supported.

    Args:
        sandbox: Handle whose `check(cwd=...)` has succeeded.
        cwd: Absolute guest working directory used for the check.
        include: Optional base-tool allowlist.
        exclude: Optional base-tool denylist.
    """
    directory = guest_cwd(cwd)
    info = sandbox.checked_info(directory)
    metadata = {item.name: item for item in info.tools}

    # A dict alone would hide duplicate names. Check both the keys and count;
    # every file tool must have exactly one guest definition.
    if set(metadata) != set(FILE_TOOL_NAMES) or len(info.tools) != len(metadata):
        raise SpaceError(
            "SPACE guest returned invalid tool metadata.", kind=SpaceErrorKind.PROTOCOL
        )

    # Build fresh core definitions to reuse schemas, configuration, and host-only
    # handlers. "/" is a construction placeholder, not the guest cwd: every
    # environment handler and its path guidance are replaced below before use.
    templates = base_harness_tools(cwd="/", include=include, exclude=exclude)
    definitions: list[HarnessToolDefinition] = []
    guidance = (
        f"Environment tools use the SPACE sandbox with working directory `{directory}`. "
        "All file and output-log paths refer to that sandbox. The working directory is not a security boundary.",
        "File-tool JSON requests and Bash command strings have a 48 KiB UTF-8 input limit. Oversized input is rejected before execution.",
    )
    for definition in templates:
        name = definition.tool.name
        if name in FILE_TOOL_NAMES:
            original = definition.tool
            assert isinstance(original, Tool)
            guest = metadata[name]

            # The host validates model arguments with its own schema. Require
            # the guest to accept that same contract before exposing the tool.
            if original.schema["parameters"] != guest.parameters_schema:
                raise SpaceError(
                    "SPACE guest tool schemas do not match the host.",
                    kind=SpaceErrorKind.PROTOCOL,
                )
            # Only transport lives here. The guest worker constructs this same
            # core file tool with the guest cwd and runs its original handler.
            handler = _FileHandler(sandbox, directory, name)
            definitions.append(
                HarnessToolDefinition(
                    tool=original.replace(
                        handler=handler,
                        retry_on_timeout=False,
                        description=guest.description,
                    ),
                    prompt_snippet=guest.prompt_snippet,
                    prompt_guidelines=(*guest.prompt_guidelines, *guidance),
                )
            )
        elif name == "bash":
            # Bash already supports executor injection. Keep its core argument
            # and result handling; the wrapper adds SPACE readiness and errors.
            bash = bash_tool(cwd=directory, executor=SpaceBashExecutor(sandbox))
            executable_bash = cast(Tool[Any, Any], bash.tool)
            definitions.append(
                replace(
                    bash,
                    tool=executable_bash.replace(
                        handler=_BashHandler(sandbox, directory),
                        retry_on_timeout=False,
                    ),
                    prompt_guidelines=(
                        "Run foreground, non-interactive Bash commands. Background jobs can keep the output stream open until the command deadline.",
                        f"Commands default to a {sandbox.default_timeout:g}-second server deadline.",
                        *guidance,
                    ),
                )
            )
        else:
            # Plan state and delegation belong to the host harness. Preserve
            # their handlers instead of forwarding them into the guest worker.
            definitions.append(definition)
    return tuple(definitions)


class _FileHandler:
    def __init__(self, sandbox: SpaceSandbox, cwd: str, name: str) -> None:
        self._sandbox = sandbox
        self._cwd = cwd
        self._name = name

    async def __call__(
        self,
        args: BaseModel,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str | ToolFailure:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError
        try:
            self._sandbox.require_checked(self._cwd)

            # Send correlation fields only. Host callbacks, credentials, and
            # live run state must not cross the worker request boundary.
            request = WorkerRequest.model_validate(
                {
                    "tool": self._name,
                    "arguments": args.model_dump(mode="json", by_alias=True),
                    "context": WorkerContext(
                        run_id=context.run_id,
                        agent_name=context.agent_name,
                        tool_call_id=context.tool_call_id,
                    ),
                }
            )
            try:
                encoded = encode_request(request)
            except ValueError:
                return _failure(
                    "SPACE file-tool request exceeds the 48 KiB input limit.",
                    "input_limit",
                )
            result = await self._sandbox.execute_worker(
                encoded,
                cwd=self._cwd,
                cancellation_token=cancellation_token,
            )
            if result.timed_out:
                # A timed-out write may have changed files. Report failure
                # without retrying it or attempting the operation on the host.
                return _failure(
                    "SPACE file tool timed out; inspect the target before retrying.",
                    "timeout",
                )
            response = decode_response(result.stdout)

            # Preserve native typed failures. Ordinary file-tool error text
            # remains text so the remote tool keeps the base tool's behavior.
            if response.error is not None:
                return ToolFailure(
                    text=response.text,
                    error=ToolError(
                        message=response.error.message, kind=response.error.kind
                    ),
                )
            return response.text
        except SpaceError as exc:
            return _failure(str(exc), str(exc.kind))
        except Exception:
            return _failure("Failed to execute SPACE file tool.", "error")


class _BashCall:
    """Retain one typed executor error across the native tool error boundary."""

    def __init__(self, sandbox: SpaceSandbox) -> None:
        self._executor = SpaceBashExecutor(sandbox)
        self.error: SpaceError | None = None

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        try:
            return await self._executor.run(request, cancellation_token)
        except SpaceError as exc:
            self.error = exc
            raise


class _BashHandler:
    def __init__(self, sandbox: SpaceSandbox, cwd: str) -> None:
        self._sandbox = sandbox
        self._cwd = cwd

    async def __call__(
        self,
        args: BaseModel,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> Any:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError
        try:
            self._sandbox.require_checked(self._cwd)
            # The base Bash schema owns these arguments. Validate its command
            # size before its handler's generic executor-error boundary.
            command = args.model_dump()["command"]
            if len(command.encode("utf-8")) > INPUT_LIMIT:
                return _failure(
                    "SPACE Bash command exceeds the 48 KiB input limit.", "input_limit"
                )
            # The native Bash handler converts executor exceptions to a generic
            # failure. Keep typed SPACE errors in this call's executor only;
            # sharing that error slot would mix results from concurrent calls.
            execution = _BashCall(self._sandbox)
            definition = bash_tool(cwd=self._cwd, executor=execution)
            tool = cast(Tool[Any, Any], definition.tool)
            result = await tool.run(args, cancellation_token, context)
            if execution.error is not None:
                return _failure(str(execution.error), str(execution.error.kind))
            return result
        except SpaceError as exc:
            return _failure(str(exc), str(exc.kind))
        except Exception:
            return _failure("Failed to execute SPACE Bash tool.", "error")


def _failure(text: str, kind: str) -> ToolFailure:
    return ToolFailure(text=text, error=ToolError(message=text, kind=kind))
