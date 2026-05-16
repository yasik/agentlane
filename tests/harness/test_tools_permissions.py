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
    parse_tool_permission_grants,
)


def _request(
    *,
    tool_name: str = "read",
    operation: ToolOperation = ToolOperation.READ_FILE,
    cwd: Path,
    path: Path | None = None,
) -> ToolPermissionRequest:
    return ToolPermissionRequest(
        tool_name=tool_name,
        operation=operation,
        cwd=cwd,
        path=path,
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
