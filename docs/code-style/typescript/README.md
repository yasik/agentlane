# TypeScript Code Style

These pages hold the TypeScript conventions referenced by `AGENTS.md`.
AgentLane's TypeScript currently lives in repository packages such as
`packages/process_bridge_ts/`: Bun-run package code, pure logic in `.ts` files,
and tests under package-local `tests/` directories.

Read the page that matches the work you are doing instead of treating
`AGENTS.md` as a long-form style manual.

## Pages

1. [Conventions](./conventions.md): general style, destructuring, imports and
   exports, variables, control flow, helper placement, comments, and protocol
   code.
2. [Testing and tooling](./testing-and-tooling.md): test approach, type
   checking, formatting, the Biome-enforced rules, and the commands that gate a
   change.
