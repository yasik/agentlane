# Tool Design

Use this page when adding or changing first-party tools, especially tools under
`agentlane.harness.tools`.

## Tool surfaces

- Build executable tools on the existing `agentlane.models.Tool` primitive.
- Harness base tools should expose a `HarnessToolDefinition`, which wraps a
  `ToolSpec` plus prompt metadata for `HarnessToolsShim`. Use executable
  `Tool` values for local function tools, and declarative `ToolSpec` values
  only when the harness runner owns execution.
- Export public helpers from `agentlane.harness.tools`. Keep implementation
  modules internal and underscore-prefixed.
- Keep construction-time configuration on the helper function. Example:
  `read_tool(cwd=workspace)` captures path policy before the model can call the
  tool.
- Register standard first-party tools through `base_harness_tools()` only after
  the tool behavior, tests, docs, and example are complete.

## Model-facing contract

- Treat the tool name, description, argument schema, prompt snippet, prompt
  guidelines, and result text as a single model-facing contract.
- The `Tool.description` belongs in the JSON tool schema. Keep it short,
  action-oriented, and explicit about important limits or defaults.
- When an established related tool exists in AgentLane or an adjacent project,
  match its description style and result shape where practical.
- Prompt snippets and guidelines belong on `HarnessToolDefinition`, not on the
  executable `Tool`.
- Use private Pydantic argument models for tool inputs. Add
  `Field(description="...")` for every model-facing field.
- Use simple, stable argument names. Prefer names that describe the model's
  intent over names from internal implementation details.

## Results

- Prefer plain text results unless a tool has a clear structured-output
  contract.
- Make successful output deterministic. Use stable labels, ordering, numbering,
  and continuation messages so models can reliably refer back to results.
- Include enough context for the model to recover or continue. Filesystem tools
  should include resolved paths when a successful result depends on path
  resolution.
- Keep truncation deterministic and visible in the result text. Use shared
  output constants such as `TEXT_MAX_LINES`, `TEXT_MAX_BYTES`,
  `BASH_MAX_LINES`, and `BASH_MAX_BYTES`.
- Apply caller-provided limits before global output caps.

## Error results

- Tool-level failures should return model-facing text, not raise out of the
  handler. This includes invalid ranges, missing paths, denied paths,
  unreadable files, unsupported file types, empty inputs, and similar
  operational failures.
- Do not build model-facing errors from `str(exc)` or interpolated exception
  objects. Exception messages can contain tracebacks, private paths, provider
  internals, or other details the model should not receive.
- Construct specific error messages from validated local context. For example,
  build `file not found: <resolved path>` from a path you control.
- Add a catch-all guard at the async handler boundary for tools that touch the
  filesystem, process execution, or external services. The guard should return
  a stable generic failure message.
- Validate simple argument constraints before side effects. This keeps errors
  predictable and avoids passing bad values into shared helpers that may raise.

## Filesystem tools

- Use `ToolPathResolver` for path resolution.
- Capture `cwd` at tool construction time. Relative paths resolve from that
  captured directory.
- Document whether absolute paths are allowed. The current Phase 11 path policy
  allows absolute paths and does not enforce a sandbox boundary.
- Prefer incremental reads and walks for large files or large directories.
- Treat binary or unsupported content as a clear tool result. For text tools,
  decode invalid UTF-8 with replacement characters when preserving surrounding
  text is useful.
- Do not shell out for behavior that can be implemented directly in Python with
  stable standard-library or project helpers.

## Implementation shape

- Keep each tool in its own internal module, such as `_read.py`.
- Define constants for the tool name, description, prompt snippet, prompt
  guideline, and generic error text.
- Keep the async handler thin. It should adapt the `Tool` call boundary, handle
  cancellation-token plumbing, and delegate tool behavior to typed helpers.
- Split validation, execution, output collection, and formatting into small
  helpers when that makes the result contract easier to test.
- Use small internal dataclasses when a helper needs to return content plus
  metadata such as continuation or error state.
- Use `del cancellation_token` or pass the token through intentionally. Avoid
  leaving unused parameters ambiguous.

## Tests

- Test the exact successful result shape.
- Test direct tool execution and normal runner tool-loop execution.
- Test prompt snippets and guidelines through `HarnessToolsShim` when the tool
  provides prompt metadata.
- Cover path resolution, caller limits, global truncation, and all expected
  operational error results.
- Add regression tests that force exception messages containing traceback-like
  or sensitive text and assert that the model-facing result is sanitized.
- For filesystem tools, include edge cases such as directories, missing files,
  denied or failed reads, binary content, invalid text bytes, offsets, and empty
  inputs.

## Documentation and examples

- Update `docs/harness/tools.md` when public tool behavior changes.
- Add or update a runnable example when introducing a new public tool or a new
  common workflow.
- Keep public docs focused on current behavior and supported boundaries. Keep
  rollout history and planning detail in `docs/plans`.
- Document safety boundaries explicitly, including lack of sandboxing,
  allowlists, or approval workflows when those boundaries apply.
