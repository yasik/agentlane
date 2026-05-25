import asyncio
from pathlib import Path

import pytest

from agentlane.harness.tools import (
    AllOfToolPermissionPolicy,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionGrantPolicy,
    ToolPermissionOutcome,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
    evaluate_tool_permission,
    parse_tool_permission_grants,
    workspace_tool_policy,
)
from agentlane.models import ToolExecutionContext


def _request(
    *,
    tool_name: str = "read",
    operation: ToolOperation = ToolOperation.READ_FILE,
    cwd: Path,
    path: Path | None = None,
    command: str | None = None,
) -> ToolPermissionRequest:
    return ToolPermissionRequest(
        tool_name=tool_name,
        operation=operation,
        cwd=cwd,
        path=path,
        command=command,
    )


def test_workspace_permission_policy_allows_paths_inside_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.txt"
    target.write_text("notes", encoding="utf-8")
    policy = WorkspaceToolPermissionPolicy(root=workspace)

    decision = policy.check(_request(cwd=workspace, path=target))

    assert decision.outcome == ToolPermissionOutcome.ALLOW


def test_evaluate_tool_permission_applies_explicit_context(
    tmp_path: Path,
) -> None:
    seen: list[ToolPermissionRequest] = []

    class RecordingPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            seen.append(request)
            return ToolPermissionDecision.allow()

    error = asyncio.run(
        evaluate_tool_permission(
            _request(cwd=tmp_path, path=tmp_path / "notes.txt"),
            policy=RecordingPolicy(),
            context=ToolExecutionContext(
                run_id="assistant-agent:session-1",
                agent_name="Reviewer",
                tool_call_id="call_1",
                metadata={"surface": "cli"},
            ),
        )
    )

    assert error is None
    assert seen[0].run_id == "assistant-agent:session-1"
    assert seen[0].agent_name == "Reviewer"
    assert seen[0].tool_call_id == "call_1"
    assert seen[0].metadata == {"surface": "cli"}


def test_evaluate_tool_permission_preserves_explicit_request_fields(
    tmp_path: Path,
) -> None:
    seen: list[ToolPermissionRequest] = []

    class RecordingPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            seen.append(request)
            return ToolPermissionDecision.allow()

    error = asyncio.run(
        evaluate_tool_permission(
            ToolPermissionRequest(
                tool_name="read",
                operation=ToolOperation.READ_FILE,
                cwd=tmp_path,
                path=tmp_path / "notes.txt",
                run_id="explicit-run",
                metadata={"surface": "app"},
            ),
            policy=RecordingPolicy(),
            context=ToolExecutionContext(
                run_id="context-run",
                agent_name="Reviewer",
                tool_call_id="call_1",
                metadata={"surface": "cli"},
            ),
        )
    )

    assert error is None
    assert seen[0] == ToolPermissionRequest(
        tool_name="read",
        operation=ToolOperation.READ_FILE,
        cwd=tmp_path,
        path=tmp_path / "notes.txt",
        run_id="explicit-run",
        agent_name="Reviewer",
        tool_call_id="call_1",
        metadata={"surface": "app"},
    )


def test_workspace_permission_policy_denies_paths_outside_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("secret", encoding="utf-8")
    policy = WorkspaceToolPermissionPolicy(root=workspace)

    decision = policy.check(_request(cwd=workspace, path=outside))

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_permission_policy_denies_symlink_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link = workspace / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    policy = WorkspaceToolPermissionPolicy(root=workspace)

    decision = policy.check(_request(cwd=workspace, path=link / "secret.txt"))

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_permission_policy_can_restrict_operations(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspaceToolPermissionPolicy(
        root=workspace,
        allowed_operations=(ToolOperation.READ_FILE,),
    )

    decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=workspace,
            path=workspace / "notes.txt",
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_permission_policy_denies_command_without_explicit_grant(
    tmp_path: Path,
) -> None:
    policy = WorkspaceToolPermissionPolicy(root=tmp_path)

    decision = policy.check(
        _request(
            tool_name="bash",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=tmp_path,
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_permission_policy_allows_command_with_explicit_grant(
    tmp_path: Path,
) -> None:
    policy = WorkspaceToolPermissionPolicy(
        root=tmp_path,
        allowed_operations=(ToolOperation.EXECUTE_COMMAND,),
    )

    decision = policy.check(
        _request(
            tool_name="bash",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=tmp_path,
        )
    )

    assert decision.outcome == ToolPermissionOutcome.ALLOW


def test_parse_tool_permission_grants_supports_tool_and_operation_entries() -> None:
    grants, invalid_entries = parse_tool_permission_grants(
        "read, write:create_file, write:overwrite_file, bash:execute_command"
    )

    assert invalid_entries == ()
    assert [grant.tool_name for grant in grants] == ["read", "write", "write", "bash"]
    assert [grant.operation for grant in grants] == [
        None,
        ToolOperation.CREATE_FILE,
        ToolOperation.OVERWRITE_FILE,
        ToolOperation.EXECUTE_COMMAND,
    ]


def test_parse_tool_permission_grants_reports_unknown_entries() -> None:
    grants, invalid_entries = parse_tool_permission_grants(
        "read, write:read_file, unknown, bash:missing"
    )

    assert [grant.tool_name for grant in grants] == ["read"]
    assert invalid_entries == ("write:read_file", "unknown", "bash:missing")


def test_tool_permission_grant_policy_checks_operation_level_grants(
    tmp_path: Path,
) -> None:
    grants, _ = parse_tool_permission_grants("write:create_file")
    policy = ToolPermissionGrantPolicy(grants)

    create_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )
    overwrite_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.OVERWRITE_FILE,
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )

    assert create_decision.outcome == ToolPermissionOutcome.ALLOW
    assert overwrite_decision.outcome == ToolPermissionOutcome.DENY


def test_all_of_tool_permission_policy_intersects_decisions(tmp_path: Path) -> None:
    grants, _ = parse_tool_permission_grants("write:create_file")
    policy = AllOfToolPermissionPolicy(
        (
            WorkspaceToolPermissionPolicy(root=tmp_path),
            ToolPermissionGrantPolicy(grants),
        )
    )
    request = _request(
        tool_name="write",
        operation=ToolOperation.CREATE_FILE,
        cwd=tmp_path,
        path=tmp_path / "notes.txt",
    )

    decision = asyncio.run(policy.check(request))

    assert decision == ToolPermissionDecision.allow()


def test_all_of_tool_permission_policy_denies_after_approval_required(
    tmp_path: Path,
) -> None:
    class RequireApprovalPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            del request
            return ToolPermissionDecision.require_approval()

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("secret", encoding="utf-8")
    policy = AllOfToolPermissionPolicy(
        (
            RequireApprovalPolicy(),
            WorkspaceToolPermissionPolicy(root=workspace),
        )
    )
    request = _request(cwd=workspace, path=outside)

    decision = asyncio.run(policy.check(request))

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_all_of_tool_permission_policy_requires_approval_after_allows(
    tmp_path: Path,
) -> None:
    class RequireApprovalPolicy:
        def check(
            self,
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            del request
            return ToolPermissionDecision.require_approval("approval needed")

    policy = AllOfToolPermissionPolicy(
        (
            WorkspaceToolPermissionPolicy(root=tmp_path),
            RequireApprovalPolicy(),
        )
    )
    request = _request(cwd=tmp_path, path=tmp_path / "notes.txt")

    decision = asyncio.run(policy.check(request))

    assert decision == ToolPermissionDecision.require_approval("approval needed")


def test_workspace_tool_policy_allows_reads_and_approval_gates_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grants, invalid_entries = parse_tool_permission_grants(
        "read, write:create_file, bash:execute_command"
    )
    policy = workspace_tool_policy(
        root=workspace,
        grants=grants,
        require_approval_for_side_effects=True,
        allow_bash_gate=True,
    )

    read_decision = asyncio.run(
        policy.check(
            _request(
                tool_name="read",
                operation=ToolOperation.READ_FILE,
                cwd=workspace,
                path=workspace / "notes.txt",
            )
        )
    )
    create_decision = asyncio.run(
        policy.check(
            _request(
                tool_name="write",
                operation=ToolOperation.CREATE_FILE,
                cwd=workspace,
                path=workspace / "notes.txt",
            )
        )
    )
    bash_decision = asyncio.run(
        policy.check(
            _request(
                tool_name="bash",
                operation=ToolOperation.EXECUTE_COMMAND,
                cwd=workspace,
                command="printf 'hello\\n'",
            )
        )
    )

    assert invalid_entries == ()
    assert read_decision == ToolPermissionDecision.allow()
    assert create_decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL
    assert bash_decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL


def test_workspace_tool_policy_still_denies_outside_paths_before_approval(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    grants, _ = parse_tool_permission_grants("write:create_file")
    policy = workspace_tool_policy(
        root=workspace,
        grants=grants,
        require_approval_for_side_effects=True,
    )

    decision = asyncio.run(
        policy.check(
            _request(
                tool_name="write",
                operation=ToolOperation.CREATE_FILE,
                cwd=workspace,
                path=outside,
            )
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_tool_policy_requires_explicit_bash_gate(
    tmp_path: Path,
) -> None:
    grants, _ = parse_tool_permission_grants("bash:execute_command")
    policy = workspace_tool_policy(
        root=tmp_path,
        grants=grants,
        require_approval_for_side_effects=True,
    )

    decision = asyncio.run(
        policy.check(
            _request(
                tool_name="bash",
                operation=ToolOperation.EXECUTE_COMMAND,
                cwd=tmp_path,
                command="printf 'hello\\n'",
            )
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY
