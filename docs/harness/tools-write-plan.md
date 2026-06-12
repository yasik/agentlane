# write_plan Tool

`plan_tool()` exposes a `write_plan` tool for creating or replacing the current
task plan.

Parameters:

1. `explanation: str | None = None`
2. `plan: list[PlanItem]`

Each plan item has:

1. `step: str`
2. `status: "pending" | "in_progress" | "completed"`

## Permissions

`write_plan` updates harness shim state only. It does not read local files,
write local files, start processes, or issue a `ToolPermissionRequest`.

Each call replaces the previous plan. Partial item updates are not part of the
current public contract. A plan must contain at least one item, each step must
contain non-whitespace text, and at most one item may be `in_progress`.

Successful model-facing tool result:

```text
Plan updated
```

The tool name and success message are exported as the public constants
`PLAN_TOOL_NAME` and `PLAN_UPDATED_MESSAGE` from `agentlane.harness.tools`, so
consumers need not mirror the literals. On a successful update the runner emits
a [`RunPlanUpdatedEvent`](runner.md#plan-updated-events) carrying the structured
plan (`RunPlanItem` tuples plus the optional `explanation`); consumers render
plan UX from that typed event instead of string-matching the success message.

The plan payload itself is intended for clients and shims to render. The tool
does not echo the full checklist back to the model after a successful update.
Invalid plan structure returns stable text such as
`plan must contain at least one item`,
`plan steps must not be empty`, or
`at most one plan step can be in_progress`. Malformed argument types are
rejected by the normal tool argument validation path.

When used through `HarnessToolsShim`, the latest plan update is persisted in
`RunState.shim_state` under `harness-tools:plan` for the default shim name.
Custom shim names use the same pattern: `{shim_name}:plan`.
