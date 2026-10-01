import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { load } from "cheerio";
import {
  docsRoot,
  docsRoute,
  resolveDocsLink,
} from "../src/lib/docs-paths.mjs";
import { prepareSearchIndex, searchDocs } from "../src/lib/docs-search.ts";

const index = JSON.parse(
  await readFile(new URL("../dist/docs/search.json", import.meta.url), "utf8"),
);
const sources = (await readdir(docsRoot, { recursive: true })).filter(
  (file) => file.endsWith(".md") && !file.startsWith("plans/"),
);
const documents = new Map(
  await Promise.all(
    index.map(async ({ href }) => [
      href,
      load(
        await readFile(
          new URL(`../dist${href}index.html`, import.meta.url),
          "utf8",
        ),
      ),
    ]),
  ),
);

test("every public root document has one route, search entry, and navigation link", () => {
  assert.ok(
    sources.includes("README.md"),
    "Root documentation index is required",
  );
  assert.deepEqual(
    index.map(({ href }) => href).sort(),
    sources.map(docsRoute).sort(),
  );
  assert.equal(new Set(index.map(({ href }) => href)).size, sources.length);
  for (const { href, title, text } of index) {
    const $ = documents.get(href);
    assert.equal($("main").length, 1);
    assert.equal($("h1").length, 1);
    assert.equal($("h1").text(), title);
    assert.ok(text.includes(title));
    assert.equal($('nav[aria-label="Documentation"] a').length, sources.length);
    assert.equal($('a[aria-current="page"]').attr("href"), href);
    assert.equal(
      new URL($('link[rel="canonical"]').attr("href")).pathname.replace(
        /\/?$/,
        "/",
      ),
      href,
    );
    assert.equal($("astro-island").length, 0);
    assert.equal($("a[href*='/docs/plans/']").length, 0);
    const ids = $("[id]")
      .map((_, el) => $(el).attr("id"))
      .get();
    assert.equal(ids.length, new Set(ids).size, `duplicate ID in ${href}`);
  }
});

test("portal links and heading fragments resolve in generated HTML", () => {
  for (const [href, $] of documents) {
    $("a[href]").each((_, el) => {
      const link = $(el).attr("href");
      if (!link.startsWith("/docs/") && !link.startsWith("#")) return;
      const url = new URL(link, `https://example.com${href}`);
      const target = documents.get(url.pathname);
      assert.ok(target, `${href} links to missing ${link}`);
      if (url.hash) {
        const id = decodeURIComponent(url.hash.slice(1));
        assert.ok(
          target("[id]")
            .toArray()
            .some((node) => target(node).attr("id") === id),
          `${href} links to missing heading ${link}`,
        );
      }
    });
  }
});

test("search finds body symbols, ranks titles, and handles empty and missing queries", () => {
  const prepared = prepareSearchIndex(index);
  assert.equal(searchDocs(prepared, "  ").length, 0);
  assert.equal(searchDocs(prepared, "unfindable-portal-term-2938").length, 0);
  assert.ok(
    searchDocs(prepared, "single_threaded_runtime").some(({ href }) =>
      href.includes("runtime/"),
    ),
  );
  assert.ok(
    searchDocs(prepared, "compaction")[0]
      .title.toLowerCase()
      .includes("compaction"),
  );
  assert.deepEqual(
    searchDocs(prepared, "HARNESS COMPACTION"),
    searchDocs(prepared, "harness compaction"),
  );
  assert.ok(searchDocs(prepared, "runtime").length <= 12);
});

test("source links preserve fragments and query strings and reject broken or internal targets", () => {
  const source = path.join(docsRoot, "harness/runner.md");
  assert.equal(
    resolveDocsLink("../process-bridge/README.md?view=full#protocol", source),
    "/docs/process-bridge/?view=full#protocol",
  );
  assert.equal(
    resolveDocsLink("../process-bridge/", source),
    "/docs/process-bridge/",
  );
  assert.equal(
    resolveDocsLink("../../README.md#quick-start", source),
    "https://github.com/yasik/agentlane/blob/main/README.md#quick-start",
  );
  assert.equal(
    resolveDocsLink("../../examples/", source),
    "https://github.com/yasik/agentlane/tree/main/examples",
  );
  assert.equal(
    resolveDocsLink("https://example.com/file.md", source),
    "https://example.com/file.md",
  );
  assert.equal(resolveDocsLink("#run-events", source), "#run-events");
  assert.throws(
    () => resolveDocsLink("missing.md", source),
    /Missing documentation link/,
  );
  assert.throws(
    () => resolveDocsLink("../plans/private.md", source),
    /internal plan/,
  );
  assert.throws(
    () => resolveDocsLink("../../../outside.md", source),
    /leaves the repository/,
  );
});

test("the sitemap covers docs and excludes plans, and homepage links reach portal pages", async () => {
  const sitemap = load(
    await readFile(new URL("../dist/sitemap-0.xml", import.meta.url), "utf8"),
    { xml: true },
  );
  const paths = sitemap("loc")
    .map((_, node) =>
      new URL(sitemap(node).text()).pathname.replace(/\/?$/, "/"),
    )
    .get();
  for (const { href } of index)
    assert.ok(paths.includes(href), `${href} missing from sitemap`);
  assert.ok(paths.every((url) => !url.includes("/plans/")));
  const home = load(
    await readFile(new URL("../dist/index.html", import.meta.url), "utf8"),
  );
  assert.ok(home('.nav a[href="/docs/"]').length > 0);
  home('a[href^="/docs/"]').each((_, el) =>
    assert.ok(documents.has(home(el).attr("href"))),
  );
});
