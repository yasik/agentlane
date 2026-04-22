# Default Agent Skills Quickstart

This example shows the first harness-native skills flow.

It keeps the core agent surface unchanged and adds a local skills library
through `SkillsShim`, which is attached via `AgentDescriptor.shims`.

The script uses:

1. a local `skills/` directory beside the example,
2. an explicit `FilesystemSkillLoader`,
3. one `SkillsShim` that exposes `activate_skill(name: str)` to the model.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_skills_quickstart/main.py
```

## What It Shows

1. local skill discovery through the default filesystem loader
2. `SkillsShim` as a normal first-party shim
3. one skills instruction block merged into the effective system instructions
4. a catalog with skill name, description, and `SKILL.md` location
5. model-driven activation through the normal tool loop
6. `<skill_content>` payloads returned through the tool loop
7. activated skill state persisted in `RunState.shim_state`
8. later turns continuing with the same skill-aware run state
