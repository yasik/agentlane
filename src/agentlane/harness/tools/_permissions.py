"""Permission primitives for first-party harness tools."""

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from agentlane.models import ToolExecutionContext


class ToolOperation(StrEnum):
    """Permission operation names for first-party local tools."""

    READ_FILE = "read_file"
    SEARCH_FILES = "search_files"
    CREATE_FILE = "create_file"
    OVERWRITE_FILE = "overwrite_file"
    MODIFY_FILE = "modify_file"
    CREATE_DIRECTORY = "create_directory"
    EXECUTE_COMMAND = "execute_command"


class ToolPermissionOutcome(StrEnum):
    """Decision outcomes returned by tool permission policies."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


def _empty_metadata() -> dict[str, object]:
    """Return a typed empty metadata mapping for permission dataclasses."""
    return {}


@dataclass(frozen=True, slots=True)
class ToolPermissionRequest:
    """Context for one permission check before a local tool operation."""

    tool_name: str
    operation: ToolOperation
    cwd: Path
    path: Path | None = None
    command: str | None = None
    skill_name: str | None = None
    reason: str | None = None
    run_id: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class ToolPermissionDecision:
    """One allow, deny, or approval-required decision."""

    outcome: ToolPermissionOutcome
    reason: str | None = None

    @classmethod
    def allow(cls) -> "ToolPermissionDecision":
        """Return an allow decision."""
        return cls(outcome=ToolPermissionOutcome.ALLOW)

    @classmethod
    def deny(cls, reason: str | None = None) -> "ToolPermissionDecision":
        """Return a deny decision."""
        return cls(outcome=ToolPermissionOutcome.DENY, reason=reason)

    @classmethod
    def require_approval(
        cls,
        reason: str | None = None,
    ) -> "ToolPermissionDecision":
        """Return an approval-required decision."""
        return cls(outcome=ToolPermissionOutcome.REQUIRE_APPROVAL, reason=reason)

    @property
    def allowed(self) -> bool:
        """Return whether the operation is allowed."""
        return self.outcome == ToolPermissionOutcome.ALLOW


type ToolPermissionCheckResult = (
    ToolPermissionDecision | Awaitable[ToolPermissionDecision]
)
type ToolApprovalCallback = Callable[
    [ToolPermissionRequest, ToolPermissionDecision],
    ToolPermissionCheckResult,
]


class ToolPermissionPolicy(Protocol):
    """Policy hook for first-party harness tool operations."""

    def check(self, request: ToolPermissionRequest) -> ToolPermissionCheckResult:
        """Return the decision for one permission request."""
        ...


class AllowAllToolPermissionPolicy:
    """Default permissive policy that preserves current tool behavior."""

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        del request
        return ToolPermissionDecision.allow()


@dataclass(frozen=True, slots=True)
class ToolPermissionGrant:
    """One whole-tool or operation-specific grant."""

    tool_name: str
    operation: ToolOperation | None = None

    def allows(self, request: ToolPermissionRequest) -> bool:
        """Return whether this grant permits the request."""
        if self.tool_name != request.tool_name:
            return False

        return self.operation is None or self.operation == request.operation


class ToolPermissionGrantPolicy:
    """Allow requests matching one of the configured operation grants."""

    def __init__(self, grants: Iterable[ToolPermissionGrant]) -> None:
        self._grants = tuple(grants)

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        if any(grant.allows(request) for grant in self._grants):
            return ToolPermissionDecision.allow()
        return ToolPermissionDecision.deny()


class AllOfToolPermissionPolicy:
    """Require every nested policy to allow a request."""

    def __init__(self, policies: Iterable[ToolPermissionPolicy]) -> None:
        self._policies = tuple(policies)

    async def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        approval_decision: ToolPermissionDecision | None = None

        for policy in self._policies:
            decision = await _resolve_permission_result(policy.check(request))

            if decision.outcome == ToolPermissionOutcome.DENY:
                return decision

            if decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL:
                approval_decision = approval_decision or decision
                continue

            if not decision.allowed:
                return decision

        if approval_decision is not None:
            return approval_decision

        return ToolPermissionDecision.allow()


def workspace_tool_policy(
    *,
    root: str | Path,
    grants: Iterable[ToolPermissionGrant] | None = None,
    require_approval_for_side_effects: bool = False,
    allow_bash_gate: bool = False,
) -> AllOfToolPermissionPolicy:
    """Build the common workspace policy used by application harnesses.

    The returned policy composes the public low-level primitives:

    1. workspace containment for path operations,
    2. optional tool or operation-level grants,
    3. optional approval gating for side effects, and
    4. optional command-execution admission so `bash` can reach the approval
       gate.
    """
    policies: list[ToolPermissionPolicy] = [
        WorkspaceToolPermissionPolicy(
            root=root,
            allowed_operations=_workspace_policy_operations(
                allow_bash_gate=allow_bash_gate,
            ),
        )
    ]
    if grants is not None:
        policies.append(ToolPermissionGrantPolicy(grants))
    if require_approval_for_side_effects:
        policies.append(_SideEffectApprovalPolicy())

    return AllOfToolPermissionPolicy(policies)


class _SideEffectApprovalPolicy:
    """Require application approval before side-effecting operations."""

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        if request.operation in _SIDE_EFFECT_OPERATIONS:
            return ToolPermissionDecision.require_approval()
        return ToolPermissionDecision.allow()


class WorkspaceToolPermissionPolicy:
    """Allow operations only when path targets stay inside a workspace root."""

    root: Path
    allowed_operations: frozenset[ToolOperation] | None

    def __init__(
        self,
        *,
        root: str | Path,
        allowed_operations: Iterable[ToolOperation | str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.allowed_operations = (
            None
            if allowed_operations is None
            else frozenset(
                _coerce_operation(operation) for operation in allowed_operations
            )
        )

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        if (
            self.allowed_operations is not None
            and request.operation not in self.allowed_operations
        ):
            return ToolPermissionDecision.deny()
        if request.operation not in _PATH_OPERATIONS:
            # Non-path ops (currently only EXECUTE_COMMAND) require an
            # explicit grant via `allowed_operations`.
            if self.allowed_operations is None:
                return ToolPermissionDecision.deny()

            return ToolPermissionDecision.allow()

        if request.path is None or not _is_path_inside_root(
            request.path, root=self.root
        ):
            return ToolPermissionDecision.deny()

        return ToolPermissionDecision.allow()


async def evaluate_tool_permission(
    request: ToolPermissionRequest,
    *,
    policy: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
    context: ToolExecutionContext | None = None,
) -> str | None:
    """Return a model-facing denial result, or None when execution may proceed."""
    active_request = _request_with_context(
        request,
        context=context,
    )
    active_policy = policy or AllowAllToolPermissionPolicy()
    decision = await _resolve_permission_result(active_policy.check(active_request))
    if (
        decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL
        and approval_callback is not None
    ):
        decision = await _resolve_permission_result(
            approval_callback(active_request, decision)
        )

    if decision.allowed:
        return None
    return format_tool_permission_result(request=active_request, decision=decision)


def format_tool_permission_result(
    *,
    request: ToolPermissionRequest,
    decision: ToolPermissionDecision,
) -> str:
    """Render a stable model-facing result for denied or approval-required calls."""
    if decision.reason is not None and decision.reason.strip() != "":
        return decision.reason

    if decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL:
        return _format_approval_required(request)
    return _format_permission_denied(request)


def parse_tool_permission_grants(
    value: str | None,
) -> tuple[tuple[ToolPermissionGrant, ...], tuple[str, ...]]:
    """Parse comma-separated whole-tool and operation-level permission grants."""
    if value is None or value.strip() == "":
        return (), ()

    grants: list[ToolPermissionGrant] = []
    invalid_entries: list[str] = []
    for raw_entry in value.split(","):
        entry = raw_entry.strip().lower()
        if entry == "":
            continue

        grant = _parse_permission_grant(entry)
        if grant is None:
            invalid_entries.append(entry)
            continue
        grants.append(grant)

    return tuple(grants), tuple(invalid_entries)


def _parse_permission_grant(entry: str) -> ToolPermissionGrant | None:
    tool_name, separator, operation_name = entry.partition(":")
    operations = _TOOL_OPERATIONS_BY_TOOL.get(tool_name)
    if operations is None:
        return None
    if separator == "":
        return ToolPermissionGrant(tool_name=tool_name)

    try:
        operation = ToolOperation(operation_name)
    except ValueError:
        return None
    if operation not in operations:
        return None
    return ToolPermissionGrant(tool_name=tool_name, operation=operation)


async def _resolve_permission_result(
    result: ToolPermissionCheckResult,
) -> ToolPermissionDecision:
    if inspect.isawaitable(result):
        return await result
    return result


def _request_with_context(
    request: ToolPermissionRequest,
    *,
    context: ToolExecutionContext | None,
) -> ToolPermissionRequest:
    """Return `request` with explicit tool execution context applied."""
    if context is None:
        return request
    return replace(
        request,
        run_id=request.run_id or context.run_id,
        agent_name=request.agent_name or context.agent_name,
        tool_call_id=request.tool_call_id or context.tool_call_id,
        metadata=(request.metadata if request.metadata else dict(context.metadata)),
    )


def _format_permission_denied(request: ToolPermissionRequest) -> str:
    subject = _permission_subject(request)
    if subject is None:
        return f"permission denied: {request.tool_name} is not allowed"
    return f"permission denied: {request.tool_name} is not allowed for {subject}"


def _format_approval_required(request: ToolPermissionRequest) -> str:
    if request.operation == ToolOperation.EXECUTE_COMMAND:
        return (
            "approval required: bash command requires application approval "
            "before execution"
        )
    subject = _permission_subject(request)
    if subject is None:
        return (
            f"approval required: {request.tool_name} requires application "
            "approval before execution"
        )
    return (
        f"approval required: {request.tool_name} requires application approval "
        f"for {subject} before execution"
    )


def _permission_subject(request: ToolPermissionRequest) -> str | None:
    if request.path is not None:
        return f"`{request.path}`"
    if request.command is not None:
        return "this command"
    return None


def _is_path_inside_root(path: Path, *, root: Path) -> bool:
    return _real_path_for_request(path).is_relative_to(_real_path(root))


def _real_path_for_request(path: Path) -> Path:
    if path.exists():
        return _real_path(path)
    nearest_parent = _nearest_existing_parent(path)
    suffix = path.relative_to(nearest_parent)
    return _real_path(nearest_parent) / suffix


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _real_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _coerce_operation(operation: ToolOperation | str) -> ToolOperation:
    if isinstance(operation, ToolOperation):
        return operation
    return ToolOperation(operation)


_PATH_OPERATIONS = frozenset(
    {
        ToolOperation.READ_FILE,
        ToolOperation.SEARCH_FILES,
        ToolOperation.CREATE_FILE,
        ToolOperation.OVERWRITE_FILE,
        ToolOperation.MODIFY_FILE,
        ToolOperation.CREATE_DIRECTORY,
    }
)

_SIDE_EFFECT_OPERATIONS = frozenset(
    {
        ToolOperation.CREATE_FILE,
        ToolOperation.OVERWRITE_FILE,
        ToolOperation.MODIFY_FILE,
        ToolOperation.CREATE_DIRECTORY,
        ToolOperation.EXECUTE_COMMAND,
    }
)

_WORKSPACE_POLICY_PATH_OPERATIONS = (
    ToolOperation.READ_FILE,
    ToolOperation.SEARCH_FILES,
    ToolOperation.CREATE_FILE,
    ToolOperation.OVERWRITE_FILE,
    ToolOperation.MODIFY_FILE,
    ToolOperation.CREATE_DIRECTORY,
)


def _workspace_policy_operations(
    *,
    allow_bash_gate: bool,
) -> tuple[ToolOperation, ...]:
    if allow_bash_gate:
        return (*_WORKSPACE_POLICY_PATH_OPERATIONS, ToolOperation.EXECUTE_COMMAND)
    return _WORKSPACE_POLICY_PATH_OPERATIONS


_TOOL_OPERATIONS_BY_TOOL: dict[str, frozenset[ToolOperation]] = {
    "read": frozenset({ToolOperation.READ_FILE}),
    "find": frozenset({ToolOperation.SEARCH_FILES}),
    "grep": frozenset({ToolOperation.READ_FILE, ToolOperation.SEARCH_FILES}),
    "write": frozenset(
        {
            ToolOperation.CREATE_FILE,
            ToolOperation.OVERWRITE_FILE,
            ToolOperation.CREATE_DIRECTORY,
        }
    ),
    "patch": frozenset({ToolOperation.MODIFY_FILE}),
    "bash": frozenset({ToolOperation.EXECUTE_COMMAND}),
}
