---
name: release
description: Prepare an AgentLane release with lockstep versioning across the root package and workspace packages, release notes based on changes since the last remote tag, full verification, and local-only git tag creation.
---

# Release

Use this skill when the user wants to cut a release, bump versions, draft
release notes, or create a git tag for PyPI or GitHub release preparation.

## Quick start

1. Run `bash .agents/skills/release/scripts/run.sh`.
2. The release must halt unless the current branch is `main`.
3. The release must halt if `git status --short` is not empty.
4. If the current versions are not in lockstep yet, update:
   - `pyproject.toml`
   - every discovered `packages/*/pyproject.toml`
5. Keep all package versions identical for a release.
6. Do not add compatibility bounds between workspace packages as part of this
   release workflow.
7. Write release notes to `docs/releases/v<version>.md` using the short
   template in `references/release_notes_template.md`.
8. Run `/usr/bin/make format`, `/usr/bin/make lint`, and `/usr/bin/make tests`.
9. Create a local annotated tag with `git tag -a v<version> -F docs/releases/v<version>.md`.
10. Do not push the tag. Leave tag pushing to the user after review.

## Versioning policy

This repository uses lockstep versioning.

Rules:

1. The root package and all workspace packages share the same release version.
2. While the project is still pre-`1.0`, treat `0.MINOR.0` as the release line
   for meaningful public changes.
3. Use `PATCH` only for safe fixes, docs, and internal improvements that do not
   intentionally break documented public APIs.
4. Use a `MINOR` bump for new public capabilities or any public behavior change
   users should adapt to.

## Release workflow

### 1. Confirm repository state

Before editing anything:

1. ensure the current branch is exactly `main`
2. ensure `git status --short` is empty
3. halt immediately if either check fails
4. do not discard unrelated changes just to force a release through

### 2. Review release context from the remote

Always base the release review on the latest remote tag, not just local tags.

Run:

```bash
bash .agents/skills/release/scripts/run.sh
```

This helper:

1. verifies the current branch is `main`
2. verifies the worktree is clean
3. fetches tags from `origin`
4. reads the current versions from the root and workspace packages
5. checks whether the repo is already in lockstep
6. finds the latest remote semver tag
7. prints the review range and the git commands to inspect commits and files

If there are no remote tags yet, treat the release as the initial release and
review the full history.

### 3. Bump versions in lockstep

Update:

1. `pyproject.toml`
2. every discovered `packages/*/pyproject.toml`

All version values should match exactly.

Do not add or change inter-package compatibility bounds as part of this
workflow.

### 4. Write release notes

Create `docs/releases/v<version>.md`.

Base the notes on:

1. `git log --reverse --no-merges --oneline <range>`
2. `git diff --name-only <range>` or `git log --name-only --pretty=format: --diff-filter=AM HEAD` for an initial release
3. user-facing API and behavior changes
4. examples and documentation added or updated

Keep the notes short. Prefer one or two sentences of summary plus a few bullets
for the most notable user-facing changes. Do not turn release notes into a full
project history dump.

Use `references/release_notes_template.md` as the starting point.

Format rules:

1. summarize the release in one or two sentences
2. keep only the most notable changes and fixes
3. group them into short sections such as `Added`, `Changed`, and `Fixed`
4. attach commit links or short commit references for each bullet
5. omit low-signal internal churn unless it materially affects users
6. do not hard-wrap or reflow the release notes to fit an 80-column line limit
7. keep each summary paragraph on one physical line
8. keep each bullet on one physical line

The template shape is intentionally short, closer to:

```markdown
# v<version>

## Summary

One or two concise sentences.

## Added

- Short user-facing addition (`abc1234`)

## Changed

- Short user-facing change (`def5678`)

## Fixed

- Short user-facing fix (`fedcba9`)
```

For an initial release, make that explicit in the summary.

The helper intentionally stops at guardrails plus release context. Do the
commit review and release-note writing directly in the release task instead of
trying to encode the whole release process in a script.

### 5. Verify

Run from the repository root:

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Do not create a release tag until all three pass.

### 6. Create the tag locally only

Create the tag:

```bash
git tag -a v<version> -F docs/releases/v<version>.md
```

Rules:

1. create the tag locally only
2. do not push the tag
3. use the release notes file as the annotated tag body
4. report the tag name and release notes path back to the user

## Expected output

A release-ready result should include:

1. the chosen version
2. confirmation that the release was cut from `main`
3. confirmation that the worktree was clean before the release started
4. confirmation that all package versions match
5. the release notes file path
6. the verification results
7. the local tag name
8. an explicit note that the tag was not pushed
