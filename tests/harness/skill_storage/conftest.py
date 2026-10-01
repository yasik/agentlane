"""Skill storage fixtures."""

import pytest

from ._reader import MemorySkillReader


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
