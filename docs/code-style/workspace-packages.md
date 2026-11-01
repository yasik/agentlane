# Workspace packages

AgentLane bundles its Python extension packages into the root `agentlane`
distribution. The root [package manifest](../../pyproject.toml) owns their
version, dependencies, and wheel contents. The TypeScript bridge has a separate
npm package and follows the [TypeScript guide](./typescript/README.md).

## Add a Python extension

Use a package for an integration with a distinct responsibility. Keep shared
framework primitives in `src/agentlane`, following the
[ownership rules](./design.md#ownership).

1. Create `packages/NAME/src/agentlane_NAME/` with `__init__.py`, `py.typed`,
   and the implementation modules. Replace `NAME` with the integration name;
   use underscores in Python import names.
2. Add a package README and tests in `packages/NAME/tests/`.
3. Add the source package path to `tool.hatch.build.targets.wheel.packages`
   in the root manifest. Add `"packages/NAME/src" = ""` to its
   `tool.hatch.build.targets.wheel.sources` table.
4. Declare required dependencies in the root `project.dependencies`. Put
   optional vendor dependencies in a named `project.optional-dependencies`
   extra and include them in the `all` extra when applicable. Keep core imports
   usable without optional dependencies.
5. Update the lockfile with `uv sync --all-extras`. Check that the package README
   explains installation of any required extra.

Bundled Python extensions do not need a separate `pyproject.toml`, a dependency
on their own distribution name, or a `workspace = true` source entry. A separately
published distribution requires an explicit packaging and release decision.

## Verify packaging

Follow the [verification workflow](../../.agents/skills/code-change-verification/SKILL.md),
then build the root distribution with the same command as the
[publish workflow](../../.github/workflows/pypi-publish.yml):

```bash
uv build --out-dir dist --no-sources
```

Inspect the wheel for the package and its `py.typed` marker. Install the wheel
in a clean environment and check the public import with the required extra.
Check that importing the core still works without that extra; an editable
checkout alone does not verify distribution contents.
