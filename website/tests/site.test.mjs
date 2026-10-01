import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import test from "node:test";
import { load } from "cheerio";
import { loadEnv } from "vite";

const env = loadEnv(process.env.NODE_ENV ?? "production", process.cwd(), "");

const html = await readFile(
  new URL("../dist/index.html", import.meta.url),
  "utf8",
);
const $ = load(html);
const canonical = new URL($('link[rel="canonical"]').attr("href"));

test("static HTML contains the full page, anchored sections, and ordered examples", () => {
  assert.equal($("main").length, 1);
  assert.equal($("h1").length, 1);
  assert.equal($("h1").text(), "# A runtime for autonomous organizations.");
  assert.equal($("#idea > p").length, 3);
  assert.equal($("#modules article").length, 4);
  assert.deepEqual(
    $("[role=tab]")
      .map((_, el) => $(el).text())
      .get(),
    ["Engineering", "Medical", "Hedge Fund"],
  );
  assert.equal($("[role=tab][aria-selected=true]").text(), "Engineering");
  assert.equal(
    $("[role=tabpanel]:not([hidden])").attr("id"),
    "example-panel-engineering",
  );
  assert.equal($("[role=tabpanel]").length, 3);
  assert.doesNotMatch(html, /organization\.example|organization available/);
  const ids = $("[id]")
    .map((_, el) => $(el).attr("id"))
    .get();
  assert.equal(ids.length, new Set(ids).size, "IDs must be unique");
  $("a[href^='#']").each((_, el) =>
    assert.ok(ids.includes($(el).attr("href").slice(1))),
  );
  $("[aria-controls]").each((_, el) =>
    assert.ok(ids.includes($(el).attr("aria-controls"))),
  );
  assert.match(
    $(".code-block pre").text(),
    /async def main\(\):\n    async with distributed_runtime\(\) as runtime:/,
  );
  assert.match($("footer").text(), /Sponsored by Diadia\./);
  assert.equal($(".sponsor").text(), "Open source. Sponsored by Diadia.");
  assert.equal($(".wordmark > .mark").length, 0);
  assert.equal($("footer span p").length, 0);
});

test("menu and heading links reach stable targets, including without JavaScript", async () => {
  assert.deepEqual(
    $(".nav a")
      .slice(0, 2)
      .map((_, el) => $(el).attr("href"))
      .get(),
    ["#idea", "#examples"],
  );
  assert.equal($("#examples astro-island").length, 1);
  const pages = [
    html,
    await readFile(new URL("../dist/404.html", import.meta.url), "utf8"),
  ];
  for (const page of pages) {
    const doc = load(page, { scriptingEnabled: false });
    const ids = doc("[id]")
      .map((_, el) => doc(el).attr("id"))
      .get();
    assert.equal(ids.length, new Set(ids).size);
    doc("h1, h2, h3").each((_, heading) => {
      const link = doc(heading).children("a.heading-anchor");
      assert.equal(link.length, 1);
      assert.ok(doc(heading).attr("id"));
      assert.equal(link.attr("href"), `#${doc(heading).attr("id")}`);
      assert.equal(link.text(), "#".repeat(Number(heading.tagName.slice(1))));
      assert.ok(link.attr("aria-label").startsWith("Link to "));
    });
  }
});

test("SEO metadata, social cards, and structured data share one canonical origin", async () => {
  assert.match($("title").text(), /^AgentLane \|/);
  assert.ok($('meta[name="description"]').attr("content").length > 80);
  assert.equal(canonical.pathname, "/");
  assert.equal(
    canonical.origin,
    new URL(env.SITE_URL || "https://getagentlane.dev").origin,
  );
  assert.equal($('meta[property="og:url"]').attr("content"), canonical.href);
  const image = new URL($('meta[property="og:image"]').attr("content"));
  assert.equal(image.origin, canonical.origin);
  assert.equal($('meta[name="twitter:image"]').attr("content"), image.href);
  await access(new URL(`../dist${image.pathname}`, import.meta.url));
  const data = JSON.parse($('script[type="application/ld+json"]').text());
  assert.equal(data["@type"], "SoftwareSourceCode");
  assert.equal(data.url, canonical.href);
  assert.equal(data.codeRepository, "https://github.com/yasik/agentlane");
  assert.equal(
    $('meta[name="robots"]').attr("content"),
    env.VERCEL_ENV === "preview" ? "noindex, nofollow" : "index, follow",
  );
});

test("sitemap, robots, and 404 are valid static outputs", async () => {
  const sitemap = await readFile(
    new URL("../dist/sitemap-0.xml", import.meta.url),
    "utf8",
  );
  const urls = load(sitemap, { xml: true })("loc")
    .map((_, el) => el.children[0].data)
    .get();
  assert.ok(urls.includes(canonical.href));
  assert.ok(
    urls.every(
      (url) =>
        url === canonical.href || new URL(url).pathname.startsWith("/docs/"),
    ),
  );
  const robots = await readFile(
    new URL("../dist/robots.txt", import.meta.url),
    "utf8",
  );
  if (env.VERCEL_ENV === "preview") assert.match(robots, /Disallow: \//);
  else
    assert.ok(
      robots.includes(`Sitemap: ${canonical.origin}/sitemap-index.xml`),
    );
  const notFound = load(
    await readFile(new URL("../dist/404.html", import.meta.url), "utf8"),
  );
  assert.equal(notFound("h1").text(), "# Page not found");
  assert.equal(
    notFound('meta[name="robots"]').attr("content"),
    "noindex, nofollow",
  );
  assert.equal(notFound("a[href='/']").length, 1);
});

test("only the example player hydrates; styles and scripts are local build assets", async () => {
  assert.equal($("astro-island").length, 1);
  assert.equal($("astro-island").attr("client"), "visible");
  const assets = $("script[src], link[rel=stylesheet]")
    .map((_, el) => $(el).attr("src") || $(el).attr("href"))
    .get();
  assert.ok(assets.length > 0);
  await Promise.all(
    assets.map(async (asset) => {
      assert.ok(asset.startsWith("/_astro/"));
      await access(new URL(`../dist${asset}`, import.meta.url));
    }),
  );
});
