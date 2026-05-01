# Default Agent Skills Quickstart

This example shows a broader harness-native skills flow paired with
`RunnerHooks`, using a clinical case-review scenario.

It keeps the core agent surface unchanged and adds a local skills library
through `SkillsShim`, which is attached via `AgentDescriptor.shims`.

The script uses:

1. a local `skills/` directory beside the example,
2. an explicit `FilesystemSkillLoader`,
3. one `SkillsShim` that exposes a cache-stable `activate_skill(name: str)`
   tool,
4. one `RunnerHooks` implementation that logs lifecycle and skill-activation
   events.

The bundled example catalog contains:

1. `acute-chest-pain`
2. `drug-reaction-triage`
3. `thyrotoxicosis-patterns`

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_skills_quickstart/main.py
```

## What It Shows

1. local discovery of several clinical-case skills through the filesystem loader
2. `SkillsShim` as a normal first-party shim
3. one skills instruction block merged into the effective system instructions
4. a catalog with skill name, description, and `SKILL.md` location
5. model-driven activation through the normal tool loop
6. `RunnerHooks` logging agent start and finish, model boundaries, and
   `activate_skill` activity
7. `<skill_content>` payloads returned through the tool loop
8. activated skill state persisted in `RunState.shim_state`
9. the final effective system instruction printed at the end of the demo
10. later turns continuing with the same multi-skill-aware run state
