---
name: docs-portal
description: Rebuild and verify the AgentLane website documentation portal after docs-sync or edits to root documentation, navigation, or portal rendering.
---

# Documentation portal

Keep public Markdown in root `docs/`. Astro reads those files directly.
`docs/plans/` is internal and must stay out of routes, navigation, and search.

## Workflow

1. If you change documentation prose, use `dev-docs` from the session's skill
   catalog. Apply its review checklist to the changed pages. Reuse a completed
   review from docs-sync when the text has not changed. If the skill is missing,
   report that limitation and do not claim a completed style review.
2. Use Node.js 24 and pnpm 10.29.3. From the repository root, install the website
   dependencies with `make -C website install` if needed.
3. Run `make -C website check`. This checks formatting and types, builds the
   website from root documentation, and tests the generated routes and links.
4. Fix failures within the changed scope and repeat the check. For layout or
   interaction changes, use `make -C website dev` to check a changed page on
   desktop and mobile. Check navigation, search, code blocks, and both themes.
5. Report the source pages, check results, and unresolved failures. Generated
   files stay in ignored `website/dist/`; do not commit them.

The build creates `/docs/` from `docs/README.md`. Other Markdown files retain
their relative path without `.md`; nested README files use directory routes.
The build derives navigation and search from the same public collection.
Use relative Markdown links in source files so they also work on GitHub.

The website workflow runs on public docs changes. Deployment uses the website's
normal release process. A local rebuild does not publish the site.

## Navigation contract

Apply these rules to all public documentation on every rebuild:

- Discover every public Markdown file under `docs/` recursively. Folder paths
  define section membership and nesting depth. Exclude `docs/plans/`.
- Keep `docs/README.md` as the documentation home. Use each top-level folder
  as a section, with its README page first if present.
- Use a nested folder's README title and route for its group. Show that page
  once. A folder without a README has a label without a link; do not create
  a page or route for it.
- Derive the sidebar, breadcrumbs, and previous/next links from the same tree.
  Every public page must appear once in navigation and remain in search.
- Use concise Markdown page titles. Omit parent names that the hierarchy
  already supplies. Group tool references under Harness > Tools and use
  each exact tool name as its title, without a "Tool" suffix. Apply title
  changes in source Markdown so pages, navigation, and search agree.
- Use `docs/README.md` to order siblings. Top-level label and order overrides
  can remain, but must not select pages or limit nesting. Do not add manual
  hierarchy lists for specific sections or providers.
- Include added folders and pages automatically, even when the documentation
  index does not list them. When moving sources, repair incoming links because
  folder paths also determine routes.

Check the generated structure against every public source path. Verify folder
ancestry, README placement, page coverage, and links across the documentation.
