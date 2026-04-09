#!/usr/bin/env python3
"""Print minimal release context and enforce release guardrails."""

import argparse
import subprocess
import tomllib
from pathlib import Path


def run_git(
    repo_root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one git command from the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def repo_root() -> Path:
    """Resolve the current repository root."""
    result = run_git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip())


def ensure_main_branch(repo_root_path: Path) -> str:
    """Ensure the release is being prepared from the main branch."""
    result = run_git(repo_root_path, "branch", "--show-current")
    branch = result.stdout.strip()
    if branch != "main":
        raise RuntimeError(
            "Release preparation must run from the main branch. "
            f"Current branch: {branch or '<detached>'}."
        )
    return branch


def ensure_clean_worktree(repo_root_path: Path) -> None:
    """Ensure the working tree is clean before release preparation starts."""
    result = run_git(repo_root_path, "status", "--short")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return

    joined = "\n".join(lines)
    raise RuntimeError(
        "Release preparation requires a clean working tree. "
        "Commit or stash changes first.\n"
        f"{joined}"
    )


def discover_version_file_paths(repo_root_path: Path) -> list[Path]:
    """Return the root package file plus all workspace package files."""
    paths: list[Path] = [Path("pyproject.toml")]
    packages_root = repo_root_path / "packages"
    if not packages_root.is_dir():
        return paths

    for package_file in sorted(packages_root.glob("*/pyproject.toml")):
        paths.append(package_file.relative_to(repo_root_path))
    return paths


def read_versions(repo_root_path: Path) -> dict[str, str]:
    """Read package versions from the root and workspace package files."""
    versions: dict[str, str] = {}
    for relative_path in discover_version_file_paths(repo_root_path):
        absolute_path = repo_root_path / relative_path
        with absolute_path.open("rb") as handle:
            data = tomllib.load(handle)
        project = data.get("project", {})
        version = project.get("version")
        if not isinstance(version, str):
            raise RuntimeError(f"Missing string version in {relative_path}.")
        versions[str(relative_path)] = version
    return versions


def latest_semver_tag(repo_root_path: Path) -> str | None:
    """Return the latest fetched semver tag, if one exists."""
    result = run_git(repo_root_path, "tag", "--sort=-v:refname")
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate.startswith("v"):
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check release guardrails and print release context."
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote used for tags. Defaults to origin.",
    )
    return parser.parse_args()


def print_release_context(
    *,
    repo_root_path: Path,
    remote: str,
    branch: str,
    versions: dict[str, str],
    last_tag: str | None,
) -> None:
    """Print the small release context that is useful to the skill."""
    print("Release context")
    print(f"Repository: {repo_root_path}")
    print(f"Remote: {remote}")
    print(f"Branch: {branch}")
    print("Worktree: clean")
    print()

    print("Version files:")
    for path, version in versions.items():
        print(f"- {path}: {version}")
    lockstep = len(set(versions.values())) == 1
    print(f"Lockstep: {'yes' if lockstep else 'no'}")
    print()

    if last_tag is None:
        review_range = "HEAD"
        print("Last remote tag: none")
        print("Review range: initial release (full history)")
    else:
        review_range = f"{last_tag}..HEAD"
        print(f"Last remote tag: {last_tag}")
        print(f"Review range: {review_range}")
    print()

    print("Suggested review commands:")
    if last_tag is None:
        print("- git log --reverse --no-merges --oneline HEAD")
        print("- git log --name-only --pretty=format: --diff-filter=AM HEAD")
    else:
        print(f"- git log --reverse --no-merges --oneline {review_range}")
        print(f"- git diff --name-only {review_range}")


def main() -> int:
    """Run the release helper."""
    args = parse_args()
    repo_root_path = repo_root()
    branch = ensure_main_branch(repo_root_path)
    ensure_clean_worktree(repo_root_path)
    run_git(repo_root_path, "fetch", "--tags", args.remote)

    versions = read_versions(repo_root_path)
    last_tag = latest_semver_tag(repo_root_path)
    print_release_context(
        repo_root_path=repo_root_path,
        remote=args.remote,
        branch=branch,
        versions=versions,
        last_tag=last_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
