import path from "node:path";
import { docsRoot, resolveDocsLink } from "./docs-paths.mjs";

/** Adds portal metadata and resolves links only for root documentation. */
export default function remarkDocs() {
  return (tree, file) => {
    const source = path.resolve(file.path);
    if (!source.startsWith(`${docsRoot}/`)) return;

    const headings = tree.children.filter(
      (node) => node.type === "heading" && node.depth === 1,
    );
    if (headings.length !== 1) throw new Error(`Expected one H1 in ${source}`);
    const description = tree.children.find((node) => node.type === "paragraph");
    const frontmatter = file.data.astro.frontmatter;
    frontmatter.docsTitle = plainText(headings[0]);
    frontmatter.docsDescription = description
      ? plainText(description).slice(0, 220)
      : frontmatter.docsTitle;
    frontmatter.docsText = plainText(tree);
    const links = [];
    const imageReferences = new Set();
    visit(tree, (node) => {
      if (node.type === "imageReference") imageReferences.add(node.identifier);
    });
    visit(tree, (node) => {
      if (["link", "definition", "image"].includes(node.type)) {
        node.url = resolveDocsLink(
          node.url,
          source,
          node.type === "image" || imageReferences.has(node.identifier),
        );
        if (node.type === "link" && node.url.startsWith("/docs/")) {
          links.push(node.url.split(/[?#]/)[0]);
        }
      }
    });
    frontmatter.docsLinks = [...new Set(links)];
  };
}

function plainText(node) {
  if (node.type === "html") return "";
  if (typeof node.value === "string") return node.value;
  return (node.children ?? [])
    .map(plainText)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function visit(node, callback) {
  callback(node);
  node.children?.forEach((child) => visit(child, callback));
}
