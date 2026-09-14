import pytest

from agentlane.harness import RunResult, RunState


def test_result_conversion_preserves_complete_state() -> None:
    state = RunState(
        instructions="stored",
        history=[{"role": "user", "content": "你好\n"}],
        responses=[],
    )
    result = RunResult(
        final_output={"answer": [None, "你好\n"]},
        responses=[],
        turn_count=2,
        run_state=state,
    )

    record = result.to_dict()

    assert record == {
        "final_output": {"answer": [None, "你好\n"]},
        "responses": [],
        "turn_count": 2,
        "run_state": {
            "instructions": "stored",
            "history": state.history,
            "responses": [],
            "shim_state": {},
            "turn_count": 0,
            "revision": 0,
        },
    }


@pytest.mark.parametrize("value", [object(), float("nan"), {1: "invalid"}])
def test_result_conversion_rejects_invalid_values(value: object) -> None:
    result = RunResult(final_output=value, responses=[], turn_count=1)
    with pytest.raises((TypeError, ValueError)):
        result.to_dict()
