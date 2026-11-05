# Query JSON with jq

This example runs the same filter against a local file and a mounted file.
Both calls use the installed jq executable and return the active record's ID.
Two printed `1` values confirm the results. The mounted file is supplied through
stdin; it does not need an operating-system mount.

Install jq on `PATH`, then run from the repository root:

```bash
uv run python examples/harness/jq_json/main.py
```

The example uses temporary local storage and calls no model or external service.
No API key is required. It uses the permissive local execution default. See the
[jq reference](../../../docs/harness/tools/jq.md) for permissions, limits, and
executor requirements.
