"""Skill storage fixtures."""

from pathlib import Path

import pytest

from ._reader import MemorySkillReader


@pytest.fixture(name="local_skill_root")
def fixture_local_skill_root(tmp_path: Path) -> Path:
    root = tmp_path / ".agents" / "skills"
    for name in ("refund", "review"):
        skill = root / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Local {name}.\n---\n"
            f"Follow the local {name} instructions.\n",
            encoding="utf-8",
        )
        (skill / "policy.md").write_text(f"Local {name} policy.\n", encoding="utf-8")

    (tmp_path / "report.txt").write_text("Workspace report.\n", encoding="utf-8")
    return root


@pytest.fixture(name="skill_reader")
def fixture_skill_reader() -> MemorySkillReader:
    return MemorySkillReader(
        {
            "skills/refund/SKILL.md": (
                b"---\nname: refund\ndescription: Handle refunds.\n"
                b"tools: read, read, write\ndisallowedTools: [bash]\n"
                b"metadata: {version: 2}\n---\nFollow the refund policy.\n"
            ),
            "skills/refund/assets/logo.bin": b"\x00\xff",
            "skills/refund/references/nested/policy.md": b"Refund within 30 days.\n",
            "skills/refund/scripts/run.py": b"print('refund')\n",
            "skills/refund/notes.md": b"Notes\n",
        }
    )
