#!/usr/bin/env python3
"""Publish an already-prepared AgentLane release."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

SEMVER_TAG_RE = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


class ReleaseError(RuntimeError):
    """Raised when the release cannot be published."""


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one command from a fixed working directory."""
    return subprocess.run(  # noqa: S603
        list(command),
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def run_git(
    repo_root_path: Path,
    *args: str,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one git command from the repository root."""
    return run_command(
        ("git", *args),
        cwd=repo_root_path,
        check=check,
        capture_output=capture_output,
    )


def repo_root() -> Path:
    """Resolve the current repository root."""
    result = run_git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip())


def ensure_command_available(command: str) -> None:
    """Ensure a required command is available."""
    if shutil.which(command) is None:
        raise ReleaseError(f"Required command is not available: {command}")


def normalize_tag(value: str) -> str:
    """Return a normalized v-prefixed semantic version tag."""
    tag = value.strip()
    if not tag:
        raise ReleaseError("Pass TAG=vX.Y.Z.")
    if not tag.startswith("v"):
        tag = f"v{tag}"
    if SEMVER_TAG_RE.fullmatch(tag) is None:
        raise ReleaseError(f"Invalid release tag: {value}")
    return tag


def tag_version(tag: str) -> str:
    """Return the changelog version label for a release tag."""
    return normalize_tag(tag).removeprefix("v")


def resolve_tag(tag: str | None) -> str:
    """Resolve the release tag to publish."""
    if tag:
        return normalize_tag(tag)
    raise ReleaseError("Pass TAG=vX.Y.Z.")


def ensure_main_branch(repo_root_path: Path) -> None:
    """Ensure publishing happens from main."""
    result = run_git(repo_root_path, "branch", "--show-current")
    branch = result.stdout.strip()
    if branch != "main":
        raise ReleaseError(
            "Release publishing must run from main. "
            f"Current branch: {branch or '<detached>'}."
        )


def ensure_local_tag_exists(repo_root_path: Path, tag: str) -> None:
    """Ensure the release tag exists locally before publishing."""
    result = run_git(
        repo_root_path,
        "rev-parse",
        "--verify",
        f"refs/tags/{tag}",
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseError(f"Local tag not found: {tag}")


def extract_changelog_entry(changelog: str, tag: str) -> str:
    """Extract the Keep a Changelog entry for a release tag."""
    version = tag_version(tag)
    entry_pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n.*?(?=^## \[|^\[[^\]]+\]:|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = entry_pattern.search(changelog)
    if match is None:
        raise ReleaseError(f"Could not find CHANGELOG.md entry for {tag}.")
    return match.group(0).strip() + "\n"


def release_notes_for_tag(repo_root_path: Path, tag: str) -> str:
    """Read the changelog entry used as the GitHub release body."""
    changelog_path = repo_root_path / "CHANGELOG.md"
    if not changelog_path.is_file():
        raise ReleaseError("CHANGELOG.md is required before publishing a release.")
    return extract_changelog_entry(changelog_path.read_text(encoding="utf-8"), tag)


def write_temporary_release_notes(body: str) -> Path:
    """Write release notes to a temporary file for the gh CLI."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="agentlane-release-",
        suffix=".md",
        delete=False,
    )
    with handle:
        handle.write(body)
    return Path(handle.name)


def ensure_github_release_absent(repo_root_path: Path, tag: str) -> None:
    """Ensure a GitHub release does not already exist for the tag."""
    existing_release = run_command(
        ("gh", "release", "view", tag),
        cwd=repo_root_path,
        check=False,
    )
    if existing_release.returncode == 0:
        raise ReleaseError(f"GitHub release already exists for {tag}.")


def prompt_publish_confirmation(tag: str, remote: str) -> bool:
    """Ask the user to confirm publishing."""
    try:
        answer = input(
            f"Publish {tag} to {remote} and create the GitHub release? "
            f"Type {tag} to continue: "
        )
    except EOFError:
        return False
    return answer.strip() == tag


def publish_release(
    repo_root_path: Path,
    *,
    remote: str,
    tag: str,
    notes_file: Path,
) -> None:
    """Push the release commit and tag, then create the GitHub release."""
    run_git(repo_root_path, "push", remote, "HEAD:main", capture_output=False)
    run_git(repo_root_path, "push", remote, tag, capture_output=False)
    run_command(
        (
            "gh",
            "release",
            "create",
            tag,
            "--verify-tag",
            "--title",
            tag,
            "--notes-file",
            str(notes_file),
        ),
        cwd=repo_root_path,
        capture_output=False,
    )


def env_value(name: str) -> str | None:
    """Read a non-empty environment value."""
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Publish an already-prepared AgentLane release."
    )
    parser.add_argument(
        "--tag",
        default=env_value("TAG"),
        help="Release tag to publish.",
    )
    parser.add_argument(
        "--remote",
        default=env_value("REMOTE") or "origin",
        help="Git remote used for publishing. Defaults to origin.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Publish an already-prepared release."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        repo_root_path = repo_root()
        tag = resolve_tag(args.tag)
        ensure_command_available("gh")
        ensure_main_branch(repo_root_path)
        ensure_local_tag_exists(repo_root_path, tag)
        release_notes = release_notes_for_tag(repo_root_path, tag)
        ensure_github_release_absent(repo_root_path, tag)

        print(f"Release tag: {tag}")
        print(f"Remote: {args.remote}")
        print("Release body: CHANGELOG.md")
        if not prompt_publish_confirmation(tag, args.remote):
            print("Publishing skipped.")
            return 0

        notes_file = write_temporary_release_notes(release_notes)
        try:
            publish_release(
                repo_root_path,
                remote=args.remote,
                tag=tag,
                notes_file=notes_file,
            )
        finally:
            notes_file.unlink(missing_ok=True)
        print(f"Published {tag}.")
        return 0
    except (ReleaseError, subprocess.CalledProcessError) as exc:
        print(f"release error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
