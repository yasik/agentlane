# Testing and Pull Requests

Use this page for test conventions and review-ready change expectations.

## Testing conventions

- Use `pytest` for unit tests.
- Use `pytest-asyncio` for async tests.
- Use `pytest-cov` for coverage when needed.
- Name tests with the pattern:
  `test_<what>_<condition>_<expected_result>`
- Mock external dependencies with `pytest-mock`.

## Fixture placement

- Prefer shared fixtures in `conftest.py`.
- Use `@pytest.fixture(name="<name>")` to avoid
  `redefined-outer-name` warnings.
- Name fixture functions as `fixture_<name>`.

Example:

```python
@pytest.fixture(name="my_fixture")
def fixture_my_fixture() -> MyType:
    return MyType()
```

## Pull request expectations

- Add tests when the change needs them.
- Update documentation when the public behavior or workflow changes.
- Run `/usr/bin/make format`.
- Run `/usr/bin/make lint`.
- Run the full test suite.
- Do not mention "Co-Authored" or "Authored By" in commit or PR text.

## Commit messages

- Keep commit messages concise.
- Use the imperative mood.
- Prefer small, focused commits.
