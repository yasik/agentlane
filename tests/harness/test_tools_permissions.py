import asyncio
from pathlib import Path

import pytest

from agentlane.harness.tools import (
    AllOfToolPermissionPolicy,
    PathScopeToolPermissionPolicy,
    SideEffectApprovalToolPermissionPolicy,
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
    policy = WorkspaceToolPermissionPolicy(workspace)

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


def test_evaluate_tool_permission_merges_request_and_context_metadata(
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
                metadata={"actor": "skill-x", "surface": "tool"},
            ),
            policy=RecordingPolicy(),
            context=ToolExecutionContext(
                run_id="context-run",
                agent_name="Reviewer",
                tool_call_id="call_1",
                metadata={"surface": "cli", "session": "local"},
            ),
        )
    )

    assert error is None
    assert seen[0].metadata == {
        "actor": "skill-x",
        "surface": "tool",
        "session": "local",
    }


def test_workspace_permission_policy_denies_paths_outside_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("secret", encoding="utf-8")
    policy = WorkspaceToolPermissionPolicy(workspace)

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
    policy = WorkspaceToolPermissionPolicy(workspace)

    decision = policy.check(_request(cwd=workspace, path=link / "secret.txt"))

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_permission_policy_can_restrict_operations(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspaceToolPermissionPolicy(
        workspace,
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
    policy = WorkspaceToolPermissionPolicy(tmp_path)

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
        tmp_path,
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


def test_path_scope_permission_policy_allows_workspace_and_outside_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside_file = tmp_path / "brief.md"
    workspace.mkdir()
    outside_file.write_text("review me", encoding="utf-8")
    policy = PathScopeToolPermissionPolicy((workspace, outside_file))

    workspace_decision = policy.check(
        _request(
            cwd=workspace,
            path=workspace / "notes.txt",
        )
    )
    outside_decision = policy.check(
        _request(
            cwd=workspace,
            path=outside_file,
        )
    )

    assert workspace_decision.outcome == ToolPermissionOutcome.ALLOW
    assert outside_decision.outcome == ToolPermissionOutcome.ALLOW


def test_path_scope_permission_policy_denies_unapproved_sibling_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    approved_file = tmp_path / "approved.md"
    sibling_file = tmp_path / "private.md"
    workspace.mkdir()
    approved_file.write_text("approved", encoding="utf-8")
    sibling_file.write_text("private", encoding="utf-8")
    policy = PathScopeToolPermissionPolicy((workspace, approved_file))

    decision = policy.check(
        _request(
            cwd=workspace,
            path=sibling_file,
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_path_scope_permission_policy_allows_outside_directory_descendants(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside_dir = tmp_path / "approved"
    workspace.mkdir()
    outside_dir.mkdir()
    nested = outside_dir / "nested" / "brief.md"
    policy = PathScopeToolPermissionPolicy((workspace, outside_dir))

    decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=workspace,
            path=nested,
        )
    )

    assert decision.outcome == ToolPermissionOutcome.ALLOW


def test_path_scope_permission_policy_limits_existing_file_to_exact_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside_file = tmp_path / "approved.md"
    workspace.mkdir()
    outside_file.write_text("approved", encoding="utf-8")
    policy = PathScopeToolPermissionPolicy((workspace, outside_file))

    decision = policy.check(
        _request(
            cwd=workspace,
            path=outside_file / "nested.md",
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_path_scope_permission_policy_allows_exact_non_existing_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    future_file = tmp_path / "approved.md"
    workspace.mkdir()
    policy = PathScopeToolPermissionPolicy((workspace, future_file))

    exact_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=workspace,
            path=future_file,
        )
    )
    child_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=workspace,
            path=future_file / "nested.md",
        )
    )

    assert exact_decision.outcome == ToolPermissionOutcome.ALLOW
    assert child_decision.outcome == ToolPermissionOutcome.DENY


def test_path_scope_permission_policy_empty_paths_denies_path_operations(
    tmp_path: Path,
) -> None:
    policy = PathScopeToolPermissionPolicy(())

    decision = policy.check(
        _request(
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_path_scope_permission_policy_can_restrict_operations(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PathScopeToolPermissionPolicy(
        (workspace,),
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


def test_path_scope_permission_policy_requires_explicit_command_grant(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    denied_policy = PathScopeToolPermissionPolicy((workspace,))
    allowed_policy = PathScopeToolPermissionPolicy(
        (workspace,),
        allowed_operations=(ToolOperation.EXECUTE_COMMAND,),
    )

    denied_decision = denied_policy.check(
        _request(
            tool_name="bash",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=workspace,
        )
    )
    allowed_decision = allowed_policy.check(
        _request(
            tool_name="bash",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=workspace,
        )
    )

    assert denied_decision.outcome == ToolPermissionOutcome.DENY
    assert allowed_decision.outcome == ToolPermissionOutcome.ALLOW


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


def test_parse_tool_permission_grants_preserves_duplicate_entries() -> None:
    grants, invalid_entries = parse_tool_permission_grants(
        "read, read, write:create_file, write:create_file"
    )

    assert invalid_entries == ()
    assert [(grant.tool_name, grant.operation) for grant in grants] == [
        ("read", None),
        ("read", None),
        ("write", ToolOperation.CREATE_FILE),
        ("write", ToolOperation.CREATE_FILE),
    ]


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
            WorkspaceToolPermissionPolicy(tmp_path),
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


def test_all_of_tool_permission_policy_allows_empty_composition(
    tmp_path: Path,
) -> None:
    policy = AllOfToolPermissionPolicy(())

    decision = asyncio.run(
        policy.check(
            _request(
                cwd=tmp_path,
                path=tmp_path / "notes.txt",
            )
        )
    )

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
            WorkspaceToolPermissionPolicy(workspace),
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
            WorkspaceToolPermissionPolicy(tmp_path),
            RequireApprovalPolicy(),
        )
    )
    request = _request(cwd=tmp_path, path=tmp_path / "notes.txt")

    decision = asyncio.run(policy.check(request))

    assert decision == ToolPermissionDecision.require_approval("approval needed")


def test_side_effect_approval_policy_requires_approval_for_writes(
    tmp_path: Path,
) -> None:
    policy = SideEffectApprovalToolPermissionPolicy()

    read_decision = policy.check(
        _request(
            tool_name="read",
            operation=ToolOperation.READ_FILE,
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )
    write_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )

    assert read_decision == ToolPermissionDecision.allow()
    assert write_decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL


def test_side_effect_approval_policy_can_require_specific_operations(
    tmp_path: Path,
) -> None:
    policy = SideEffectApprovalToolPermissionPolicy((ToolOperation.EXECUTE_COMMAND,))

    write_decision = policy.check(
        _request(
            tool_name="write",
            operation=ToolOperation.CREATE_FILE,
            cwd=tmp_path,
            path=tmp_path / "notes.txt",
        )
    )
    bash_decision = policy.check(
        _request(
            tool_name="bash",
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=tmp_path,
            command="printf 'hello\\n'",
        )
    )

    assert write_decision == ToolPermissionDecision.allow()
    assert bash_decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL


def test_workspace_tool_policy_allows_reads_and_requires_approval_for_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grants, invalid_entries = parse_tool_permission_grants(
        "read, write:create_file, bash:execute_command"
    )
    policy = workspace_tool_policy(
        workspace,
        grants=grants,
        require_approval_for_side_effects=True,
        require_bash_approval=True,
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


def test_workspace_tool_policy_without_grants_uses_workspace_only(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = workspace_tool_policy(workspace)

    decision = asyncio.run(
        policy.check(
            _request(
                tool_name="read",
                operation=ToolOperation.READ_FILE,
                cwd=workspace,
                path=workspace / "notes.txt",
            )
        )
    )

    assert decision == ToolPermissionDecision.allow()


def test_workspace_tool_policy_with_empty_grants_denies_all_granted_tools(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = workspace_tool_policy(workspace, grants=())

    decision = asyncio.run(
        policy.check(
            _request(
                tool_name="read",
                operation=ToolOperation.READ_FILE,
                cwd=workspace,
                path=workspace / "notes.txt",
            )
        )
    )

    assert decision.outcome == ToolPermissionOutcome.DENY


def test_workspace_tool_policy_still_denies_outside_paths_before_approval(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    grants, _ = parse_tool_permission_grants("write:create_file")
    policy = workspace_tool_policy(
        workspace,
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


def test_workspace_tool_policy_denies_bash_without_required_approval(
    tmp_path: Path,
) -> None:
    grants, _ = parse_tool_permission_grants("bash:execute_command")
    policy = workspace_tool_policy(
        tmp_path,
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


def test_workspace_tool_policy_requires_bash_approval_without_other_side_effects(
    tmp_path: Path,
) -> None:
    grants, _ = parse_tool_permission_grants("bash:execute_command")
    policy = workspace_tool_policy(
        tmp_path,
        grants=grants,
        require_bash_approval=True,
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

    assert decision.outcome == ToolPermissionOutcome.REQUIRE_APPROVAL
