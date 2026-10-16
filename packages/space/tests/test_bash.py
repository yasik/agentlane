"""Check bash behavior through the public executor and native transport."""

import shlex
from pathlib import Path

import pytest
from agentlane_space import SpaceBashExecutor, SpaceError, SpaceErrorKind
from conftest import LocalSpace, sandbox_for

from agentlane.harness.tools import BASH_MAX_BYTES, BashExecutionRequest
from agentlane.runtime import CancellationToken


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7, 124, 125])
async def test_command_preserves_output_and_exit_status(
    local_space: LocalSpace, tmp_path: Path, exit_code: int
) -> None:
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        result = await SpaceBashExecutor(sandbox).run(
            BashExecutionRequest(
                command=f"printf first; printf second >&2; exit {exit_code}",
                cwd=tmp_path,
            ),
            CancellationToken(),
        )

    assert result.output.text == "firstsecond"
    assert result.exit_code == exit_code
    assert not result.timed_out
    assert not result.output_truncated
    assert result.full_output_path is None


@pytest.mark.asyncio
async def test_command_preserves_quoting_cwd_and_environment(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    cwd = tmp_path / "guest ' workspace"
    cwd.mkdir()
    marker = tmp_path / "must-not-exist"
    value = f"quoted '\"; $(touch {shlex.quote(str(marker))})\nsecond line"
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(cwd))
        result = await SpaceBashExecutor(sandbox).run(
            BashExecutionRequest(
                command=f"printf '%s\\n' {shlex.quote(value)}; printf '%s' \"$EXAMPLE\"; pwd",
                cwd=cwd,
                env={"EXAMPLE": "environment; "},
            ),
            CancellationToken(),
        )

    assert result.exit_code == 0
    assert result.output.text == f"{value}\nenvironment; {cwd}\n"
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "large_output", ["printf '%060000d' 0", "yes line | head -2500"]
)
async def test_truncated_result_keeps_complete_private_guest_log(
    local_space: LocalSpace, tmp_path: Path, large_output: str
) -> None:
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        result = await SpaceBashExecutor(sandbox).run(
            BashExecutionRequest(
                command=f"printf 'first\\n'; {large_output}; printf '\\nlast\\n' >&2",
                cwd=tmp_path,
            ),
            CancellationToken(),
        )

    assert result.exit_code == 0
    assert result.output_truncated
    assert len(result.output.text.encode()) <= BASH_MAX_BYTES
    assert len(result.output.text.splitlines()) <= 2000
    assert result.output.text.endswith("\nlast\n")
    assert result.full_output_path is not None
    log = result.full_output_path
    assert log.read_bytes().startswith(b"first\n")
    assert log.read_bytes().endswith(result.output.text.encode())
    assert log.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_capture_failure_is_reported_and_allows_next_command(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        executor = SpaceBashExecutor(sandbox)
        with pytest.raises(SpaceError) as failure:
            await executor.run(
                BashExecutionRequest(
                    command="printf output",
                    cwd=tmp_path,
                    env={"PATH": str(empty_bin)},
                ),
                CancellationToken(),
            )
        assert failure.value.kind is SpaceErrorKind.PROTOCOL
        assert not sandbox.needs_recovery
        result = await executor.run(
            BashExecutionRequest(command="printf ready", cwd=tmp_path),
            CancellationToken(),
        )
        assert result.exit_code == 0
        assert result.output.text == "ready"
