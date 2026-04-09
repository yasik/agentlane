---
name: docs-sync
description: Review the current AgentLane code and public docs under docs/ to find missing, outdated, or misplaced documentation, then propose or apply concise documentation updates that match the repository’s structure and writing style. Ignore internal planning docs under docs/plans/.
---

# Docs Sync

Use this skill when the user wants to audit documentation coverage, keep
`docs/` aligned with implementation changes, or update public technical docs.

## Scope

This skill is for public documentation under `docs/**`.

Rules:

1. Treat `docs/plans/**` as internal tracking, not public documentation.
2. English only.
3. Keep edits concise and easy to maintain.

## Workflow

### 1. Define the scope to review

Pick one of these modes:

1. full public-doc audit against the current branch
2. doc sync for a specific feature or directory
3. diff-based audit against `main` for in-flight changes

Prefer diff-based review when there is active branch work. Prefer a full audit
 when the user asks for broad documentation cleanup.

### 2. Build a feature inventory from code

Review the selected scope in code first.

Focus on user-facing behavior:

1. public exports
2. runtime behaviors and guarantees
3. configuration options and defaults
4. examples that define supported usage
5. important developer workflows
6. renamed, removed, or newly added features

Capture lightweight evidence:

1. file path
2. symbol or setting name
3. short note about the behavior that should be documented

### 3. Review the existing docs hierarchy

Use the current docs tree as the source of structure.

Start with:

1. `docs/README.md`
2. the relevant page under `docs/runtime/`, `docs/messaging/`, `docs/harness/`,
   or `docs/transport/`

When reviewing, prefer updating an existing page over creating a new one unless
the topic clearly has no home.

If you add, remove, or rename a public docs page, update `docs/README.md`.

### 4. Compare docs against code

Look for:

1. missing public behavior
2. outdated names, defaults, or semantics
3. examples in docs that no longer match implementation
4. information that belongs on a different page for discoverability
5. public docs that accidentally expose internal rollout or planning details

### 5. Produce a Docs Sync Report

Before editing, produce a short report with:

1. doc gaps
2. inaccurate or outdated content
3. structural suggestions, if any
4. proposed file-level edits

Keep the report concise.

### 6. Apply updates if asked

When applying edits:

1. edit only the relevant public docs under `docs/**`
2. keep the current writing style and section structure
3. keep examples short and directly tied to the repo
4. prefer current-state documentation over rollout history
5. do not add jokes, slang, rhetoric, puns, or contrastive-negation phrasing

## Writing Style

Match the existing documentation style in this repository.

Rules:

1. write in plain technical English
2. prefer short sections and short paragraphs
3. use direct statements over persuasive or dramatic framing
4. explain the current behavior, boundaries, and intended usage
5. keep examples minimal and realistic
6. avoid figures of speech, jokes, slang, and rhetorical phrasing
7. avoid contrastive negation patterns such as “not X, but Y”
8. keep tone neutral and open-source appropriate

## Public Docs Boundary

For this repository, public docs currently live under `docs/**`.

Internal tracking docs live under `docs/plans/**` and should not be treated as
the public documentation surface.

## Output format

Use this report shape when auditing before edits:

Docs Sync Report

- Scope
  - What was reviewed
- Missing or weak coverage
  - Doc file or missing page -> evidence -> proposed change
- Incorrect or outdated content
  - Doc file -> issue -> correct behavior -> evidence
- Structural suggestions
  - Proposed move/add/remove -> rationale
- Proposed edits
  - File -> short summary

## References

- `references/doc-coverage-checklist.md`
