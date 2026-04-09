# Doc Coverage Checklist

Use this checklist when reviewing AgentLane public documentation.

## Review targets

- Public docs under `docs/**`
- Public docs index in `docs/README.md`
- Current implementation under `src/`
- Public examples when they define supported usage

Do not treat `docs/plans/**` as public documentation.

## Code-first pass

- Identify public exports, runtime behaviors, configuration, and examples.
- Note renamed, removed, or newly added features.
- Record short evidence only: file path plus symbol or setting.

## Doc-first pass

- Review the relevant page in `docs/`.
- Check whether the page reflects the current behavior and naming.
- Prefer updating existing pages over creating new ones.
- Update `docs/README.md` if public docs pages are added, removed, or renamed.

## Common gaps

- Missing public behavior after a new feature lands
- Old names or defaults that no longer match code
- Examples that no longer match the actual API
- Public docs that still contain rollout notes, phases, dates, or status markers
- Information placed on the wrong page for discoverability

## Writing guardrails

- Keep docs concise.
- Use plain technical English.
- Avoid slang, jokes, rhetoric, figures of speech, and puns.
- Avoid contrastive negation phrasing.
- Describe current behavior rather than implementation history.

## Patch guidance

- Keep edits narrow and local to the affected pages.
- Preserve the current section style and hierarchy.
- Prefer short examples over long explanatory blocks.
- Keep internal planning notes in `docs/plans/**`, not in public docs.
