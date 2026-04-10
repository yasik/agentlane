# Workspace Packages

Follow this process when adding a new utility package to the workspace.

## 1. Create the package structure

```bash
mkdir -p packages/<package-name>/src/agentlane_<package-name>
touch packages/<package-name>/src/agentlane_<package-name>/__init__.py
touch packages/<package-name>/src/agentlane_<package-name>/py.typed
```

## 2. Create `pyproject.toml`

Create `packages/<package-name>/pyproject.toml`:

```toml
[project]
name = "agentlane-<package-name>"
version = "0.1.0"
description = "Brief description of the package"
requires-python = ">=3.12"
dependencies = [
  # Add your dependencies here
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentlane_<package-name>"]

[tool.hatch.build.targets.wheel.sources]
"src" = ""
```

## 3. Register the package in the workspace

Update the root `pyproject.toml`:

```toml
[project]
dependencies = [
  # ... existing dependencies
  "agentlane-<package-name>",
]

[tool.uv.sources]
agentlane-<package-name> = { workspace = true }
```

## 4. Install and verify

```bash
uv sync
uv run python -c "import agentlane_<package-name>; print('Package installed')"
```

## Package guidelines

- Use `agentlane_<name>` for imports and `agentlane-<name>` for package names.
- Always use the `src/` layout:
  `packages/<name>/src/agentlane_<name>/`
- Include the `py.typed` marker.
- Place tests in `packages/<package-name>/tests/` or alongside the source when
  that is the established local pattern.
- Declare only direct dependencies in each package.
