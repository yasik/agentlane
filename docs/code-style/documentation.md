# Maintain documentation

Use this guide when you change AgentLane behavior or public documentation.
The root `docs/` directory is the source for both repository readers and the
website documentation portal.

## Before you begin

You need the following tools:

- A full repository checkout.
- Node.js 24 and pnpm 10.29.3.
- The `dev-docs` skill in your coding agent's skill catalog.

## Update the documentation

Follow this workflow after a code change:

1. Use the repository's `docs-sync` skill to compare the changed code with
   public documentation. Check real exports, defaults, and examples before
   you describe them.
2. Edit the relevant Markdown files in root `docs/`. If you add, remove, or
   rename a page, update the documentation index.
3. Apply the `dev-docs` review checklist to the changed pages. Use direct
   instructions, sentence case, descriptive links, and tested examples.
   Follow repository terminology and Simplified Technical English.
4. Use the repository's `docs-portal` skill to rebuild and check the website.
   It runs `make -C website check` from the repository root.

The `docs-sync` skill calls the portal workflow after it applies documentation
updates. A read-only audit produces a report without a rebuild. These skills
run in your coding agent; the website build does not perform an AI style review.

## Source and route rules

The portal applies the following rules during the build:

- `docs/README.md` becomes `/docs/`.
- Other Markdown files keep their relative path without the `.md` extension.
- A nested `README.md` becomes its directory route. For example,
  `docs/process-bridge/README.md` becomes `/docs/process-bridge/`.
- Each page has one level-one heading. That heading supplies the page title.
- Relative links to public docs become portal links. Links to source code and
  examples point to GitHub.
- `docs/plans/` stays outside public pages, navigation, and search.

Keep relative Markdown links in the source. Do not add portal URLs to replace
links that repository readers use. Do not copy docs into `website/src/content/`
or commit generated files from `website/dist/`.

## Build checks

The website workflow checks public documentation changes in pull requests and
on `main`. It builds production and preview output, then checks page coverage,
internal links, heading anchors, and search. A successful local build does not
deploy the website.

For build commands and hosting setup, see the
[website build instructions](../../website/README.md).
