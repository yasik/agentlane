# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.8.0]: https://github.com/yasik/agentlane/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/yasik/agentlane/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/yasik/agentlane/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/yasik/agentlane/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/yasik/agentlane/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/yasik/agentlane/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/yasik/agentlane/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yasik/agentlane/releases/tag/v0.3.0
