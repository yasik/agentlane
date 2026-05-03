---
name: release
description: Prepare and publish an AgentLane release with lockstep versioning, generated Keep a Changelog release notes, verification, a release commit, an annotated tag, and a GitHub release.
---

# Release

Use this skill when the user wants to cut an AgentLane release.

This skill is the release entrypoint. It owns the full workflow: inspect the
repository state, choose the target version, update version files, generate the
`CHANGELOG.md` entry from commits, verify the change, create the release commit,
create the annotated tag, and publish through the skill-local helper.

## Versioning Policy

AgentLane uses lockstep versioning.

Rules:

1. The root package and all discovered workspace packages must share the same
   release version.
2. While the project is pre-`1.0`, use `0.MINOR.0` for meaningful public
   changes.
3. Use `PATCH` for safe fixes, docs, and internal improvements that do not
   intentionally change documented public behavior.
4. Use `MINOR` for new public capabilities or public behavior changes users
   should notice.

## Workflow

### 1. Guardrails

Before editing:

1. confirm the current branch is exactly `main`
2. confirm `git status --short` is empty
3. fetch tags from `origin`
4. confirm local `main` matches `origin/main`
5. stop immediately if any guardrail fails

Useful commands:

```bash
git branch --show-current
git status --short
git fetch --tags origin
git rev-parse HEAD
git rev-parse origin/main
```

### 2. Release Context

Base the release review on the latest remote semver tag.

Find the latest remote tag with:

```bash
git ls-remote --tags --refs origin 'refs/tags/v*'
```

Review the range:

```bash
git log --reverse --no-merges --oneline <last-tag>..HEAD
git diff --name-only <last-tag>..HEAD
```

If there are no remote semver tags, review the initial history:

```bash
git log --reverse --no-merges --oneline HEAD
git log --name-only --pretty=format: --diff-filter=AM HEAD
```

### 3. Version Files

Read versions from:

1. `pyproject.toml`
2. any workspace package `pyproject.toml` files discovered under `packages/`

All discovered versions must match before release preparation.

Apply the target version to every discovered version file. Do not change
inter-package dependency bounds as part of release preparation.

### 4. Changelog

Generate the release entry from:

1. non-merge commits in the review range
2. changed files in the review range
3. user-facing API and behavior changes
4. examples and documentation changes that affect users

Edit `CHANGELOG.md` directly.

Format rules:

1. insert or replace `## [<version>] - <YYYY-MM-DD>` near the top
2. start with one short summary paragraph
3. use Keep a Changelog sections: `Added`, `Changed`, `Deprecated`, `Removed`,
   `Fixed`, and `Security`
4. include only sections that have content
5. attach commit links or short commit references for each bullet
6. omit low-signal internal churn unless it materially affects users
7. update the compare link at the bottom
8. keep each summary paragraph on one physical line
9. keep each bullet on one physical line

Use `references/release_notes_template.md` as the body template.

### 5. Verify

Run:

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Do not commit or tag until all three pass.

### 6. Commit And Tag

Commit only the files changed for the release:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: v<version>"
```

If workspace package version files exist, add those exact paths before
committing.

Create an annotated tag using the new `CHANGELOG.md` entry as the tag message:

```bash
git tag -a v<version> -F <temporary-release-notes-file>
```

### 7. Publish

Publish only after the release commit and annotated tag exist locally:

```bash
bash .agents/skills/release/scripts/run.sh --tag v<version>
```

The helper:

1. requires an explicit `--tag v<version>`
2. verifies the local tag exists
3. reads the GitHub release body from the matching `CHANGELOG.md` entry
4. verifies a GitHub release does not already exist for the tag
5. asks for explicit confirmation
6. pushes `HEAD:main` to `origin`
7. pushes the tag to `origin`
8. creates the GitHub release with `gh release create`

If confirmation is declined, publishing is skipped and no remote changes are
made.

## Completion Report

Report:

1. target version and tag
2. release range reviewed
3. files changed
4. verification results
5. release commit
6. whether publishing was confirmed or skipped
