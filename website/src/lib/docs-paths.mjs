import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const repositoryRoot = fileURLToPath(
  new URL("../../../", import.meta.url),
);
export const docsRoot = path.join(repositoryRoot, "docs");
export const repositoryUrl = "https://github.com/yasik/agentlane";

/** Maps a path relative to docs/ to its public route. */
export function docsRoute(source) {
  const slug = source.replace(/\.md$/, "").replace(/(^|\/)README$/, "");
  return `/docs/${slug ? `${slug.replace(/\/$/, "")}/` : ""}`;
}

/** Resolves repository Markdown links without changing the source files. */
export function resolveDocsLink(url, source, image = false) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(url)) return url;

  const [, pathname, suffix = ""] = url.match(/^([^?#]*)(.*)$/);
  if (!pathname) return url;
  const target = pathname.startsWith("/")
    ? path.resolve(repositoryRoot, `.${decodeURIComponent(pathname)}`)
    : path.resolve(path.dirname(source), decodeURIComponent(pathname));
  const relative = path.relative(repositoryRoot, target);
  if (relative.startsWith("../") || path.isAbsolute(relative)) {
    throw new Error(
      `Documentation link leaves the repository: ${url} in ${source}`,
    );
  }
  if (relative === "docs/plans" || relative.startsWith("docs/plans/")) {
    throw new Error(`Public documentation links to an internal plan: ${url}`);
  }
  if (!existsSync(target)) {
    throw new Error(`Missing documentation link target: ${url} in ${source}`);
  }
  const directory = statSync(target).isDirectory();
  const markdown = directory ? path.join(target, "README.md") : target;
  if (
    !image &&
    markdown.startsWith(`${docsRoot}/`) &&
    markdown.endsWith(".md") &&
    existsSync(markdown)
  ) {
    return `${docsRoute(path.relative(docsRoot, markdown))}${suffix}`;
  }
  const encoded = relative.split(path.sep).map(encodeURIComponent).join("/");
  if (image)
    return `https://raw.githubusercontent.com/yasik/agentlane/main/${encoded}${suffix}`;
  return `${repositoryUrl}/${directory ? "tree" : "blob"}/main/${encoded}${suffix}`;
}
