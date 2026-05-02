---
name: release
description: Prepare and publish an AgentLane release with lockstep versioning, generated Keep a Changelog release notes, verification, a release commit, an annotated tag, and a GitHub release.
---

# Release

Use this skill when the user wants to cut a release, bump versions, draft
release notes, create a release tag, or publish a GitHub release.

The skill owns the full release workflow: choosing the version, editing version
files, generating the Keep a Changelog entry from commits, verifying the
change, committing, tagging, and publishing. The only script involved is the
skill-local publish helper, which pushes the already-prepared release commit and
tag, then calls `gh release create`.

Do not create new `docs/releases/v<version>.md` files. `CHANGELOG.md` is the
release-notes source of truth, and its version entry is used as the annotated
tag message and GitHub release body.

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

## Workflow

### 1. Confirm repository state

Before editing anything:

1. ensure the current branch is exactly `main`
2. ensure `git status --short` is empty
3. fetch tags from `origin`
4. ensure local `main` matches `origin/main`
5. halt immediately if any guardrail fails
6. do not discard unrelated changes just to force a release through

Useful commands:

```bash
git branch --show-current
git status --short
git fetch --tags origin
git rev-parse HEAD
git rev-parse origin/main
```

### 2. Review release context

Always base the release review on the latest remote semver tag, not just local
tags.

Use the latest remote `vX.Y.Z` tag as the base. If there are no remote semver
tags yet, treat the release as the initial release and review the full history.

Review:

```bash
git log --reverse --no-merges --oneline <last-tag>..HEAD
git diff --name-only <last-tag>..HEAD
```

For an initial release, use:

```bash
git log --reverse --no-merges --oneline HEAD
git log --name-only --pretty=format: --diff-filter=AM HEAD
```

### 3. Choose and apply the version

Read the root package and any discovered `packages/*/pyproject.toml` version
values. All version values must match exactly before release preparation.

Rules:

1. use an explicit user-provided version when given
2. otherwise apply the versioning policy from this skill
3. update `pyproject.toml`
4. update every discovered `packages/*/pyproject.toml`
5. do not add or change inter-package compatibility bounds as part of this
   workflow

### 4. Generate the changelog entry

Generate the release entry from:

1. the non-merge commits in the review range
2. the changed files in the review range
3. user-facing API and behavior changes
4. examples and documentation added or updated

Keep the notes concise. Prefer one short summary paragraph plus a few bullets
for the most notable user-facing changes. Do not turn release notes into a full
project history dump.

Use `references/release_notes_template.md` as the starting point.

Format rules:

1. insert or replace `## [<version>] - <YYYY-MM-DD>` near the top of
   `CHANGELOG.md`
2. start the body with one short summary paragraph
3. group bullets under Keep a Changelog sections: `Added`, `Changed`,
   `Deprecated`, `Removed`, `Fixed`, and `Security`
4. include only sections that have content
5. attach commit links or short commit references for each bullet
6. omit low-signal internal churn unless it materially affects users
7. update the compare link at the bottom of `CHANGELOG.md`
8. do not hard-wrap or reflow the release notes to fit an 80-column line limit
9. keep each summary paragraph on one physical line
10. keep each bullet on one physical line

### 5. Verify

Run from the repository root:

```bash
/usr/bin/make format
/usr/bin/make lint
/usr/bin/make tests
```

Do not commit or tag the release until all three pass.

### 6. Commit and tag locally

Commit only the release files:

```bash
git add pyproject.toml packages/*/pyproject.toml CHANGELOG.md
git commit -m "release: v<version>"
```

If there are no `packages/*/pyproject.toml` files, omit that path.

Create the annotated tag from the new `CHANGELOG.md` entry. Use a temporary
file outside the repository if needed:

```bash
git tag -a v<version> -F <temporary-release-notes-file>
```

### 7. Publish

Publish only after the release commit and annotated tag already exist locally:

```bash
bash .agents/skills/release/scripts/run.sh --tag v<version>
```

The publish command:

1. verifies the local tag exists
2. reads the GitHub release body from the matching `CHANGELOG.md` entry
3. verifies a GitHub release does not already exist for the tag
4. asks for explicit confirmation
5. pushes the current `main` commit to `origin`
6. pushes the tag to `origin`
7. creates the GitHub release with `gh release create`

If confirmation is declined, publishing is skipped and no remote changes are
made.

## Expected output

A release-ready result should include:

1. the chosen version
2. confirmation that the release was cut from `main`
3. confirmation that the worktree was clean before release edits started
4. confirmation that all package versions match
5. confirmation that `CHANGELOG.md` was updated
6. verification results
7. the local release commit and tag name
8. whether publishing was confirmed or skipped
