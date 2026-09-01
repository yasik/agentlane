# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2026-09-01

AgentLane `0.13.0` adds the first persistent-agent workflow and the first cross-harness coworker integration: applications can now preserve a harness agent across process restarts and address a Claude Agent SDK participant through the normal AgentLane runtime.

### Added

- Added versioned `AgentSnapshot` values so applications can export committed `DefaultAgent` state as portable JSON and restore it at the same logical `AgentId` ([`60a0824`](https://github.com/yasik/agentlane/commit/60a0824), [`4fc8211`](https://github.com/yasik/agentlane/commit/4fc8211), [`d004bf9`](https://github.com/yasik/agentlane/commit/d004bf9))
- Added `state_path=`, `StateStore`, and `JsonFileStateStore` so default agents can restore conversation and harness state after a process restart, save successful primary runs atomically, and reject stale revisions ([`9426935`](https://github.com/yasik/agentlane/commit/9426935))
- Added the optional `agentlane[claude-agent-sdk]` integration and a runnable coworker example so native AgentLane agents can send addressed tasks to a fresh Claude Agent SDK session and use its result in the original run ([`ba197e9`](https://github.com/yasik/agentlane/commit/ba197e9), [`d078f03`](https://github.com/yasik/agentlane/commit/d078f03), [`a7e0949`](https://github.com/yasik/agentlane/commit/a7e0949))

### Changed

- Unified terminal, streaming, event-streaming, and handoff execution around shared runner paths so lifecycle fixes remain consistent across public run modes while their intentional retry behavior remains unchanged ([`14f7a19`](https://github.com/yasik/agentlane/commit/14f7a19))
- Expanded the harness documentation and examples with persistent-agent and external-coworker setup, storage boundaries, worker requirements, credential guidance, and current interoperability limits ([`9426935`](https://github.com/yasik/agentlane/commit/9426935), [`a7e0949`](https://github.com/yasik/agentlane/commit/a7e0949), [`3da5822`](https://github.com/yasik/agentlane/commit/3da5822))

### Fixed

- Made npm publishing recoverable by installing the required Python tooling, supporting exact-tag manual runs, and skipping `@agentlanejs/process-bridge` versions that are already published ([`65698a5`](https://github.com/yasik/agentlane/commit/65698a5), [`0b7c402`](https://github.com/yasik/agentlane/commit/0b7c402))
- Rejected typed and raw Claude SDK session-continuation options so each addressed task honors the documented fresh-session boundary ([`5d67cff`](https://github.com/yasik/agentlane/commit/5d67cff))

## [0.12.0] - 2026-07-07

AgentLane `0.12.0` expands the harness into a fuller app platform: markdown agent definitions, resilient context compaction, and Python/TypeScript process-bridge APIs now give host applications a typed path for configuring, observing, and controlling local AgentLane sessions.

### Added

- Added markdown agent definitions with `descriptor_from_markdown(...)`, `DefaultAgent.from_markdown(...)`, model resolution, sub-agent links, handoff support, and `disallowedTools` filtering for file-backed harness agents ([`d2a6159`](https://github.com/yasik/agentlane/commit/d2a6159), [`1fbebc7`](https://github.com/yasik/agentlane/commit/1fbebc7), [`4466dcb`](https://github.com/yasik/agentlane/commit/4466dcb))
- Added public harness compaction contracts, the stock compaction shim, and the default summary-plus-tail compactor so long-running harness agents can compact history without crashing the run when compaction fails ([`8490f76`](https://github.com/yasik/agentlane/commit/8490f76), [`aad9810`](https://github.com/yasik/agentlane/commit/aad9810), [`fabd917`](https://github.com/yasik/agentlane/commit/fabd917), [`89f24be`](https://github.com/yasik/agentlane/commit/89f24be))
- Added Python and TypeScript process bridge packages for NDJSON command/event streaming, plus the app-facing `createAgentSession(...)` API, backend entrypoint, typed callbacks, text streaming helpers, approval controls, and stdio example updates ([`a7c3f56`](https://github.com/yasik/agentlane/commit/a7c3f56), [`afe7eb4`](https://github.com/yasik/agentlane/commit/afe7eb4), [`f550f84`](https://github.com/yasik/agentlane/commit/f550f84))
- Added runtime configuration to the process bridge so TypeScript hosts can update model settings through typed session controls and receive authoritative config events from the Python backend ([`7dae388`](https://github.com/yasik/agentlane/commit/7dae388))
- Added token usage to the `llm_end` bridge event and the matching `TokenUsage` TypeScript decoder so host telemetry can show provider token totals without parsing model output ([`89f24be`](https://github.com/yasik/agentlane/commit/89f24be))

### Changed

- Published `@agentlanejs/process-bridge` as a real npm package with dist exports, declarations, package-local checks, and a release-triggered npm publish workflow ([`5bb4cb1`](https://github.com/yasik/agentlane/commit/5bb4cb1))
- Removed skill-relative tool wrapping from the skills API. `SkillsShim` now keeps workspace tools' path semantics unchanged and emits absolute paths for bundled skill resources in the activation payload ([`5c5399c`](https://github.com/yasik/agentlane/commit/5c5399c))
- Moved process-bridge documentation into `docs/process-bridge/` and refreshed TypeScript/Python code-style guidance around strict protocol decoding, extension contracts, and package verification ([`7dae388`](https://github.com/yasik/agentlane/commit/7dae388), [`28ebe61`](https://github.com/yasik/agentlane/commit/28ebe61))

### Fixed

- Hardened process-bridge protocol edge cases around startup stdout hygiene, prompt cancellation, strict decode errors, approval-policy failures, and typed approval request payloads ([`5dfa260`](https://github.com/yasik/agentlane/commit/5dfa260), [`896867b`](https://github.com/yasik/agentlane/commit/896867b), [`f550f84`](https://github.com/yasik/agentlane/commit/f550f84))
- Fixed the Claude streaming tool-thinking example and LiteLLM lock metadata for the current dependency set ([`9097700`](https://github.com/yasik/agentlane/commit/9097700))
- Restored the `structlog` dependency so clean CI installs include the runtime import surface checked by pyright ([`ebc2e26`](https://github.com/yasik/agentlane/commit/ebc2e26))

## [0.11.0] - 2026-06-14

AgentLane `0.11.0` makes harness runs easier for host applications to observe and control: tool calls now have typed success/failure outcomes, plan updates and delegation show up as structured run events, tools can read live run state, skills can resolve bundled resources by relative paths, and permission helpers cover app tools and network egress.

### Added

- Added structured tool-result primitives (`ToolError`, `ToolFailure`, `ToolOutcome`, `tool_outcome`) and `RunToolEndEvent.ok` / `error` so host UIs can distinguish failed tool calls without parsing model-facing text ([`6a75f4d`](https://github.com/yasik/agentlane/commit/6a75f4d), [`0587ff5`](https://github.com/yasik/agentlane/commit/0587ff5), [`cbeb50b`](https://github.com/yasik/agentlane/commit/cbeb50b))
- Added `RunPlanUpdatedEvent`, `RunPlanItem`, `PLAN_TOOL_NAME`, `PLAN_UPDATED_MESSAGE`, and structured `PlanUpdate` results for first-party plan tool updates ([`e73f922`](https://github.com/yasik/agentlane/commit/e73f922), [`626ef78`](https://github.com/yasik/agentlane/commit/626ef78))
- Added run-event lineage fields and `is_delegation` tagging for agent-as-tool and handoff calls so consumers can aggregate root-run telemetry correctly ([`626ef78`](https://github.com/yasik/agentlane/commit/626ef78), [`04c7f39`](https://github.com/yasik/agentlane/commit/04c7f39))
- Added live run-state access on `ToolExecutionContext.run_state`, `LiveRunStateView`, and `RunStateView` so tools can inspect task identity, shim state, and active skills during execution ([`3e37a0a`](https://github.com/yasik/agentlane/commit/3e37a0a), [`82690ac`](https://github.com/yasik/agentlane/commit/82690ac), [`ca129e5`](https://github.com/yasik/agentlane/commit/ca129e5))
- Added skill catalog sharing, active-skill accessors, and `SkillRelativePathShim` / `resolve_skill_relative_path` so active skills can reference bundled resources through `read`, `grep`, and `find` without duplicate discovery ([`7df701c`](https://github.com/yasik/agentlane/commit/7df701c), [`496f392`](https://github.com/yasik/agentlane/commit/496f392), [`398f532`](https://github.com/yasik/agentlane/commit/398f532))
- Added provider payload helpers `ReasoningPhase`, `UsageTotals`, `get_reasoning_phase`, and `get_usage_totals` for typed access to Responses reasoning phases and token totals ([`5423ea7`](https://github.com/yasik/agentlane/commit/5423ea7), [`5a83e3f`](https://github.com/yasik/agentlane/commit/5a83e3f))

### Changed

- Made `PromptTemplate` and `MultiPartPromptTemplate` default to plain string output and exported `render_instruction_text` for shared system-instruction rendering ([`5e82f83`](https://github.com/yasik/agentlane/commit/5e82f83))
- Added `Tool.handler`, `Tool.formatter`, `Tool.replace`, and `Tool.with_handler` plus `DelegatingShim` / `DelegatingBoundShim` to make wrapper shims and tools preserve future framework fields ([`d5b62ca`](https://github.com/yasik/agentlane/commit/d5b62ca), [`3e37a0a`](https://github.com/yasik/agentlane/commit/3e37a0a), [`398f532`](https://github.com/yasik/agentlane/commit/398f532))
- Exposed `BASE_TOOL_NAMES` and `extra_names` support in `base_harness_tools(...)` so applications can apply shared include/exclude selectors across base and app tools ([`cd595aa`](https://github.com/yasik/agentlane/commit/cd595aa))
- Expanded permission policies with `ToolOperation.NETWORK_ACCESS`, app-tool grant parsing, non-path operation admission, and grant-aware side-effect approval downgrades for trusted host operations ([`0db0e44`](https://github.com/yasik/agentlane/commit/0db0e44), [`fc68915`](https://github.com/yasik/agentlane/commit/fc68915), [`6fe8d92`](https://github.com/yasik/agentlane/commit/6fe8d92))
- Refreshed docs for run events, structured tool outcomes, skills, permission composition, prompt templates, model helpers, and tool design ([`7b615c9`](https://github.com/yasik/agentlane/commit/7b615c9), [`073ca37`](https://github.com/yasik/agentlane/commit/073ca37), [`4712b41`](https://github.com/yasik/agentlane/commit/4712b41), [`3ef299d`](https://github.com/yasik/agentlane/commit/3ef299d))

### Fixed

- Fixed structured failure reporting for bash timeouts, cancelled commands, crashes, and non-zero exits so run events carry typed errors while model-facing text stays unchanged ([`0587ff5`](https://github.com/yasik/agentlane/commit/0587ff5), [`cbeb50b`](https://github.com/yasik/agentlane/commit/cbeb50b))
- Fixed permission grant wildcard expansion and prevented disallowed tools from being re-added to the active context by skill/tool filtering ([`6fe8d92`](https://github.com/yasik/agentlane/commit/6fe8d92), [`ba70df8`](https://github.com/yasik/agentlane/commit/ba70df8))
- Fixed active-skill state key drift and made shim state exposed through live run-state views read-only ([`496f392`](https://github.com/yasik/agentlane/commit/496f392), [`ca129e5`](https://github.com/yasik/agentlane/commit/ca129e5))
- Fixed unrecognized provider reasoning phases to degrade to `None` instead of raising ([`5a83e3f`](https://github.com/yasik/agentlane/commit/5a83e3f))
- Fixed pre-existing mypy errors and a contradicting immediate-decision approval test that blocked the verification gate ([`a7dccc6`](https://github.com/yasik/agentlane/commit/a7dccc6), [`5d8ecd7`](https://github.com/yasik/agentlane/commit/5d8ecd7))

## [0.10.0] - 2026-06-11

AgentLane `0.10.0` makes tracing across model calls and tool execution consistent: harness runs now keep tool spans under the active generation span, provider clients reuse propagated tracing context, and documentation better explains the harness, runtime, serialization, and tracing surfaces.

### Added

- Added `Model.tracing`, `parent_span`, and `cancellation_token` propagation to the public model contract so harness runners and provider clients can coordinate tracing and cancellation across model and tool calls ([`ef14aa4`](https://github.com/yasik/agentlane/commit/ef14aa4), [`3b65ec0`](https://github.com/yasik/agentlane/commit/3b65ec0), [`cd75ffd`](https://github.com/yasik/agentlane/commit/cd75ffd))

### Changed

- Scoped harness generation spans across full runs so repeated model turns accumulate usage on the generation span and tool spans nest under the model request that triggered them ([`ef14aa4`](https://github.com/yasik/agentlane/commit/ef14aa4), [`3b65ec0`](https://github.com/yasik/agentlane/commit/3b65ec0))
- Updated the OpenAI Responses and LiteLLM provider clients to record model metadata and accumulated usage or cost onto caller-owned parent spans when one is supplied ([`3b65ec0`](https://github.com/yasik/agentlane/commit/3b65ec0), [`cd75ffd`](https://github.com/yasik/agentlane/commit/cd75ffd))
- Refreshed the README and harness, runtime, messaging, serialization, model, and tracing docs around the current public APIs, run events, tool permissions, and distributed execution model ([`5520e8a`](https://github.com/yasik/agentlane/commit/5520e8a), [`ac03193`](https://github.com/yasik/agentlane/commit/ac03193))

### Fixed

- Fixed tool execution tracing so function spans follow the active model tracing mode instead of a stale executor-level setting ([`ef14aa4`](https://github.com/yasik/agentlane/commit/ef14aa4), [`cd75ffd`](https://github.com/yasik/agentlane/commit/cd75ffd))

## [0.9.0] - 2026-06-03

AgentLane `0.9.0` turns the harness into a more practical host-app surface: generic sub-agents, first-party tool permission policies, approval/run-event streaming, and distributed example docs are now paired with runtime message identity and cancellation improvements.

### Added

- Added the generic spawned `agent` tool with inherited base tools, depth/thread guards, policy modes, and sanitized delegation failures ([`b3e62ec`](https://github.com/yasik/agentlane/commit/b3e62ec), [`fe60246`](https://github.com/yasik/agentlane/commit/fe60246), [`f55558a`](https://github.com/yasik/agentlane/commit/f55558a), [`f38eb6d`](https://github.com/yasik/agentlane/commit/f38eb6d), [`8186b08`](https://github.com/yasik/agentlane/commit/8186b08))
- Added first-party tool permission grants, workspace/path-scope policies, approval-required decisions, approval events, and docs for composing local file and command access ([`8bd7353`](https://github.com/yasik/agentlane/commit/8bd7353), [`92f9f42`](https://github.com/yasik/agentlane/commit/92f9f42), [`a26092c`](https://github.com/yasik/agentlane/commit/a26092c), [`4daa279`](https://github.com/yasik/agentlane/commit/4daa279), [`fb75688`](https://github.com/yasik/agentlane/commit/fb75688), [`c6a335d`](https://github.com/yasik/agentlane/commit/c6a335d))
- Added high-level harness run events and approval broker lifecycle support so hosts can observe run state, tool calls, approvals, and stream teardown through one event surface ([`c5a7e91`](https://github.com/yasik/agentlane/commit/c5a7e91), [`1ccc46d`](https://github.com/yasik/agentlane/commit/1ccc46d))
- Added distributed harness agents documentation and a multi-process clinical inbox copilot example that demonstrates host/worker execution from the harness layer ([`57b5b31`](https://github.com/yasik/agentlane/commit/57b5b31), [`eeba50f`](https://github.com/yasik/agentlane/commit/eeba50f), [`b3ab07f`](https://github.com/yasik/agentlane/commit/b3ab07f))

### Changed

- Exported `AgentFactory` from the runtime public API and moved `MessageContext` to `agentlane.runtime` for clearer runtime-facing imports ([`689ac51`](https://github.com/yasik/agentlane/commit/689ac51), [`d33c59d`](https://github.com/yasik/agentlane/commit/d33c59d))
- Let runtime callers provide a `message_id` to `send_message(...)` and propagated the resolved id through rejection outcomes and envelopes ([`a9d0224`](https://github.com/yasik/agentlane/commit/a9d0224), [`e48259a`](https://github.com/yasik/agentlane/commit/e48259a))
- Split first-party harness tool docs into focused pages and refined README/doc examples around current distributed and tool-permission behavior ([`5d2b14a`](https://github.com/yasik/agentlane/commit/5d2b14a), [`4d9e72d`](https://github.com/yasik/agentlane/commit/4d9e72d), [`e862772`](https://github.com/yasik/agentlane/commit/e862772), [`927461b`](https://github.com/yasik/agentlane/commit/927461b))

### Fixed

- Fixed `MessageContext` cancellation propagation and covered it across in-process and cross-worker runtime paths ([`43f95ac`](https://github.com/yasik/agentlane/commit/43f95ac), [`6e50131`](https://github.com/yasik/agentlane/commit/6e50131), [`ad0221b`](https://github.com/yasik/agentlane/commit/ad0221b), [`6596ecd`](https://github.com/yasik/agentlane/commit/6596ecd))
- Fixed the read base tool's model-facing output and tightened path resolution, allow-all policy boundaries, grep/search docs, and related tool docs consistency ([`5d69a9c`](https://github.com/yasik/agentlane/commit/5d69a9c), [`bf2704b`](https://github.com/yasik/agentlane/commit/bf2704b), [`35047ac`](https://github.com/yasik/agentlane/commit/35047ac), [`fd44746`](https://github.com/yasik/agentlane/commit/fd44746), [`d43400a`](https://github.com/yasik/agentlane/commit/d43400a))

## [0.8.0] - 2026-05-02

AgentLane `0.8.0` expands the first-party harness base tools with patch editing and bash execution, then tightens their model-facing output and examples. This release also fixes repeated skill activation so active skills stay stable across turns.

### Added

- Added the harness patch tool for applying model-authored file edits, with docs, tests, and quickstart coverage ([`6e420dd`](https://github.com/yasik/agentlane/commit/6e420dd))
- Added the harness bash tool and executor for controlled command execution from base-tool agents ([`126a114`](https://github.com/yasik/agentlane/commit/126a114), [`59a57d2`](https://github.com/yasik/agentlane/commit/59a57d2))
- Added GitHub Actions CI coverage for the repository test workflow ([`00afe15`](https://github.com/yasik/agentlane/commit/00afe15), [`fa1083d`](https://github.com/yasik/agentlane/commit/fa1083d))

### Changed

- Rebuilt the examples and README coverage around more practical harness, model, and runtime use cases ([`81ffd92`](https://github.com/yasik/agentlane/commit/81ffd92), [`76c464b`](https://github.com/yasik/agentlane/commit/76c464b))
- Slimmed bash tool output and refined base-tool streaming examples so model-visible output stays focused on actionable results ([`1bec31f`](https://github.com/yasik/agentlane/commit/1bec31f), [`a08ddfc`](https://github.com/yasik/agentlane/commit/a08ddfc), [`eb4eceb`](https://github.com/yasik/agentlane/commit/eb4eceb))

### Fixed

- Fixed repeated skill activation so an already-active skill is tracked through run state and handled without reshaping the tool schema ([`1897b21`](https://github.com/yasik/agentlane/commit/1897b21))
- Tightened bash executor typing and review fixes around the base-tool command surface ([`31170f9`](https://github.com/yasik/agentlane/commit/31170f9), [`13a6779`](https://github.com/yasik/agentlane/commit/13a6779))

## [0.7.0] - 2026-04-27

AgentLane `0.7.0` adds the extensibility foundation for higher-level harness agents. This release introduces shims, skills, and first-party base tools so `DefaultAgent` can be used as a practical starting point for agents that shape context, expose capabilities, and operate over local project files.

### Added

- Added the harness shim system for composing agent behavior around run state, prompts, tools, and lifecycle integration ([`04bfed0`](https://github.com/yasik/agentlane/commit/04bfed0), [`ed6b5e7`](https://github.com/yasik/agentlane/commit/ed6b5e7), [`2eb3d26`](https://github.com/yasik/agentlane/commit/2eb3d26))
- Added skills support with filesystem discovery, skill parsing, activation, prompt rendering, lifecycle integration, and clinical quickstart examples that demonstrate skill-loaded context and hooks ([`c875a73`](https://github.com/yasik/agentlane/commit/c875a73), [`9e6ceda`](https://github.com/yasik/agentlane/commit/9e6ceda), [`310202e`](https://github.com/yasik/agentlane/commit/310202e))
- Added first-party harness tools for reading, writing, planning, finding files, and grepping file contents, with shared prompt metadata, truncation policy, `.gitignore` handling, examples, and docs ([`a522bce`](https://github.com/yasik/agentlane/commit/a522bce), [`6a41532`](https://github.com/yasik/agentlane/commit/6a41532), [`bd4a575`](https://github.com/yasik/agentlane/commit/bd4a575), [`3b5c4d4`](https://github.com/yasik/agentlane/commit/3b5c4d4), [`9f211c0`](https://github.com/yasik/agentlane/commit/9f211c0), [`5f54a38`](https://github.com/yasik/agentlane/commit/5f54a38))

### Changed

- Expanded harness documentation around shims, skills, base tools, default agents, architecture, and code style for building opinionated higher-level agents ([`1a9ae2a`](https://github.com/yasik/agentlane/commit/1a9ae2a), [`fb751dc`](https://github.com/yasik/agentlane/commit/fb751dc))
- Generalized hook integration so shims and skills can contribute hooks alongside developer-provided hooks without constraining hooks to observation-only behavior ([`c84a7eb`](https://github.com/yasik/agentlane/commit/c84a7eb), [`1be5dc2`](https://github.com/yasik/agentlane/commit/1be5dc2))

### Fixed

- Hardened native grep behavior for invalid file types, warning output, binary files, truncation, and test organization around each tool's public surface ([`bdc3d07`](https://github.com/yasik/agentlane/commit/bdc3d07), [`cc2cdaf`](https://github.com/yasik/agentlane/commit/cc2cdaf))

## [0.6.1] - 2026-04-16

AgentLane `0.6.1` is a patch release with one end-to-end harness demo and one runtime fix. It adds a richer streamed clinical inbox copilot example and fixes runtime handler validation so string payload annotations resolve correctly.

### Added

- Added a new `clinical_inbox_copilot` harness demo that combines `DefaultAgent.run_stream(...)`, tool calls, runtime fan-out to parallel specialist agents, and a live dashboard-style clinician workflow ([`7de1afc`](https://github.com/yasik/agentlane/commit/7de1afc))

### Changed

- Updated the harness examples index and the clinical demo README so the new end-to-end workflow is easier to discover and run ([`7de1afc`](https://github.com/yasik/agentlane/commit/7de1afc), [`68f470e`](https://github.com/yasik/agentlane/commit/68f470e))

### Fixed

- Fixed runtime `@on_message` payload validation so handlers using string annotations are resolved correctly at registration time ([`7de1afc`](https://github.com/yasik/agentlane/commit/7de1afc))

## [0.6.0] - 2026-04-15

AgentLane `0.6.0` adds first-class streaming across the models and harness layers. This release introduces provider-grounded stream events for OpenAI and LiteLLM-backed providers, plus `DefaultAgent.run_stream(...)` and runnable examples that show tool calls, handoffs, and delegated agents in a streamed flow.

### Added

- Added provider-grounded streaming to the shared models contract with `Model.stream_response(...)`, `ModelStreamEvent`, native OpenAI Responses API event streaming, and LiteLLM chunk streaming support ([`20a3e49`](https://github.com/yasik/agentlane/commit/20a3e49), [`52ab0a1`](https://github.com/yasik/agentlane/commit/52ab0a1))
- Added harness-level streaming with `RunStream`, runner and lifecycle streaming paths, and `DefaultAgent.run_stream(...)` ([`52ab0a1`](https://github.com/yasik/agentlane/commit/52ab0a1), [`2ef3bb2`](https://github.com/yasik/agentlane/commit/2ef3bb2))
- Added runnable streaming examples for OpenAI reasoning/preambles, Claude thinking blocks, high-level harness streaming, and a streamed tool plus agent-as-tool plus handoff flow ([`35d546d`](https://github.com/yasik/agentlane/commit/35d546d), [`2ef3bb2`](https://github.com/yasik/agentlane/commit/2ef3bb2), [`6bd5b01`](https://github.com/yasik/agentlane/commit/6bd5b01))

### Changed

- Refined the root README, harness docs, and models docs so the streaming behavior, lower-level agent surface, and provider-specific event fidelity are easier to understand from the public documentation ([`a6dc3f6`](https://github.com/yasik/agentlane/commit/a6dc3f6), [`4ee4b67`](https://github.com/yasik/agentlane/commit/4ee4b67), [`36c6fac`](https://github.com/yasik/agentlane/commit/36c6fac))

### Fixed

- Cleaned up the streaming implementation and examples so the high-level demos show real provider reasoning and preambles, and the orchestration examples exercise the intended tool, delegation, and handoff paths reliably ([`f1b0454`](https://github.com/yasik/agentlane/commit/f1b0454), [`36c6fac`](https://github.com/yasik/agentlane/commit/36c6fac))

## [0.5.0] - 2026-04-13

AgentLane `0.5.0` changes the packaging and release model for optional integrations. The framework now publishes a single `agentlane` distribution to PyPI and exposes Braintrust and LiteLLM support through install extras instead of separate addon projects.

### Changed

- Switched the optional integration packaging model from multiple PyPI projects to one `agentlane` distribution that bundles `agentlane_braintrust`, `agentlane_litellm`, and `agentlane_openai`.
- Added `agentlane[braintrust]`, `agentlane[litellm]`, `agentlane[openai]`, and `agentlane[all]` extras so optional integrations install from the root package.
- Simplified the GitHub Actions PyPI workflow to build and publish only the root `agentlane` artifacts.

### Fixed

- Removed the trusted-publisher failure mode where addon package uploads could be rejected because the corresponding PyPI projects did not already exist.
- Updated the installation and release documentation to match the single-project publish flow.

## [0.4.1] - 2026-04-13

AgentLane `0.4.1` is a release workflow patch that fixes the trusted publishing workflow after the `0.4.0` release.

### Fixed

- Fixed the Astral setup action version used by the PyPI publishing workflow ([`aabbd3d`](https://github.com/yasik/agentlane/commit/aabbd3d))
- Updated the package versions and release metadata for the `0.4.1` patch release ([`21209bb`](https://github.com/yasik/agentlane/commit/21209bb))

## [0.4.0] - 2026-04-13

AgentLane `0.4.0` adds a higher-level stateful harness agent API centered on `DefaultAgent`, plus the docs, examples, and release automation needed to ship and publish it more cleanly.

### Added

- Added the new high-level harness agent surface with `DefaultAgent`, `AgentBase`, persisted primary-line runs, branch execution via `fork(...)`, and explicit state reset support ([`c903e24`](https://github.com/yasik/agentlane/commit/c903e248de32f53cb59d20e3f8b44310cd337735), [`5768b8d`](https://github.com/yasik/agentlane/commit/5768b8d0bd0a5486b116dba84ac119434769b5f8), [`b8b79df`](https://github.com/yasik/agentlane/commit/b8b79dfd4c84e3b31c6ac7af088e84fb84e2a684))
- Added default-agent documentation, tests, and a runnable quickstart example for the new harness entry point ([`c903e24`](https://github.com/yasik/agentlane/commit/c903e248de32f53cb59d20e3f8b44310cd337735), [`f4dfd3a`](https://github.com/yasik/agentlane/commit/f4dfd3a012ea71c66f0e4a8b17273621e7b3845d))
- Added a GitHub Actions workflow for trusted PyPI publishing across the root package and workspace packages ([`a726d45`](https://github.com/yasik/agentlane/commit/a726d45625ed18806af256be2bc04d4db6a29337))

### Changed

- Refined the public README and docs index so the runtime, models, and harness entry points are easier to navigate from the repository root ([`d5f3770`](https://github.com/yasik/agentlane/commit/d5f3770a412f16c1b9a541427e064e368fe25a79), [`5a19df3`](https://github.com/yasik/agentlane/commit/5a19df37e00ccf8cfaa2f55a32e3afcc7bcdb138), [`c1293b0`](https://github.com/yasik/agentlane/commit/c1293b045f5de902e188e6d054dfe1b5281ee50c), [`0dd493d`](https://github.com/yasik/agentlane/commit/0dd493db763c415cbc71d7095e959f42607d905c))
- Formalized the local release workflow with release-note driven tagging and GitHub release creation guidance ([`e423b5a`](https://github.com/yasik/agentlane/commit/e423b5a9e59a9a32e1528a1e9b11398cca052891))

### Fixed

- Tightened release note formatting so annotated tags and release bodies stay concise and consistent ([`dfff2af`](https://github.com/yasik/agentlane/commit/dfff2af6d29df31b14a6efd751f92d5bf935a900))

## [0.3.0] - 2026-04-09

AgentLane `0.3.0` is the initial public release. It ships the runtime and distributed execution model, the models and tracing foundations, and the first agent harness with tools, handoffs, agent-as-tool delegation, and runnable examples.

### Added

- Runtime messaging, transport serialization, and distributed host/worker execution ([`d3c6666`](https://github.com/yasik/agentlane/commit/d3c6666f7fbc72812fb7538643f1c03f416fbcbe))
- `agentlane.models`, `agentlane.tracing`, and provider packages for OpenAI, LiteLLM, and Braintrust ([`9801612`](https://github.com/yasik/agentlane/commit/9801612787b12d16e72745110bb33332e5f0d836), [`5a6b228`](https://github.com/yasik/agentlane/commit/5a6b228c8d81831443fb9a9fb36ddb8bfe4cfc05))
- Harness primitives with `Task`, `Agent`, `Runner`, first-class handoff, and agent-as-tool delegation ([`bb5af11`](https://github.com/yasik/agentlane/commit/bb5af11e0d21423dc2f2d47f30077013cecf32bf), [`9fc6e97`](https://github.com/yasik/agentlane/commit/9fc6e9770d85fa8555606b4e3dd98ef2fda6bc0f))

### Changed

- Tool ergonomics now support inferred schemas from typed functions and the `@as_tool` decorator ([`8496cb5`](https://github.com/yasik/agentlane/commit/8496cb5900c63df5802142ab1baef769b81ca510), [`bb083b0`](https://github.com/yasik/agentlane/commit/bb083b0dcb97b3d5c89cda30ae922cf3398d40a7))
- Added concise runtime and harness examples together with public docs aligned to the current architecture ([`7230931`](https://github.com/yasik/agentlane/commit/723093132bd9349c9bd5d807c4d4cd68224f372c), [`6b583a0`](https://github.com/yasik/agentlane/commit/6b583a0311e608809599013b9ccaae5aa61651e0), [`477159a`](https://github.com/yasik/agentlane/commit/477159ac68f2311d7350a9af9761878fce115101))

### Fixed

- Final pre-release cleanup removed dead code and added repo-level `vulture` configuration for ongoing dead-code checks ([`f009e5d`](https://github.com/yasik/agentlane/commit/f009e5d523a84d3e6747329522582d3196906534))

[Unreleased]: https://github.com/yasik/agentlane/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/yasik/agentlane/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/yasik/agentlane/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/yasik/agentlane/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/yasik/agentlane/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/yasik/agentlane/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/yasik/agentlane/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/yasik/agentlane/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/yasik/agentlane/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/yasik/agentlane/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/yasik/agentlane/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/yasik/agentlane/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/yasik/agentlane/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yasik/agentlane/releases/tag/v0.3.0
