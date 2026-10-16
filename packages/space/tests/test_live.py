"""Optional live check against an application-owned, prepared SPACE sandbox."""

import os
import runpy
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("SPACE_RUN_LIVE_TEST") != "1",
    reason="Set SPACE_RUN_LIVE_TEST=1 and SPACE connection values to test a live sandbox.",
)
def test_live_space_tools() -> None:
    example = Path(__file__).resolve().parents[3] / "examples/harness/space_sandbox.py"
    runpy.run_path(str(example), run_name="__main__")
