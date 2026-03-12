# Runtime TODO

## Distributed Runtime

[ ] Evaluate replacing the transport-only wire dataclasses with Pydantic models to reduce `from_json`/`to_json` and manual validation boilerplate in [distributed-runtime-architecture.md](./distributed-runtime-architecture.md). Keep the core runtime and messaging primitives as plain dataclasses unless there is a stronger reason to move those too.
