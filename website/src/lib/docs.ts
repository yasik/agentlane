import { getCollection, type CollectionEntry } from "astro:content";
import { z } from "astro/zod";
import { docsRoute } from "./docs-paths.mjs";

// These labels and ordering preferences do not limit which folders appear.
const groups: Record<string, string> = {
  start: "Start here",
  runtime: "Runtime",
  messaging: "Messaging",
  transport: "Transport",
  models: "Models",
  harness: "Harness",
  "process-bridge": "Process bridge",
  tracing: "Tracing",
  "code-style": "Contribute",
};

type DocPage = {
  entry: CollectionEntry<"docs">;
  title: string;
  description: string;
  text: string;
  href: string;
  group: string;
  order: number;
};

/** A section, a folder overview, or a page in the documentation tree. */
export type DocNavigationItem = {
  title: string;
  page?: DocPage;
  children: DocNavigationItem[];
};

const metadata = z.object({
  docsTitle: z.string().min(1),
  docsDescription: z.string().min(1),
  docsText: z.string(),
  docsLinks: z.array(z.string()),
});

/** Builds navigation and search from the same public collection as page routes. */
export async function getDocs(): Promise<DocPage[]> {
  const entries = await getCollection("docs");
  const home = entries.find((entry) => entry.id === "README");
  if (!home)
    throw new Error(
      "The portal requires root docs/README.md. Use a full repository checkout.",
    );
  const links = metadata.parse(home.rendered?.metadata?.frontmatter).docsLinks;
  const pageOrder = (href: string): number => {
    const index = links.indexOf(href);
    return index < 0 ? links.length : index;
  };
  const docs = entries
    .map((entry) => {
      const data = metadata.parse(entry.rendered?.metadata?.frontmatter);
      const group = entry.id.includes("/") ? entry.id.split("/")[0] : "start";
      const groupIndex = Object.keys(groups).indexOf(group);
      return {
        entry,
        title: data.docsTitle,
        description: data.docsDescription,
        text: data.docsText,
        href: docsRoute(`${entry.id}.md`),
        group: groups[group] ?? folderTitle(group),
        order: groupIndex < 0 ? Object.keys(groups).length : groupIndex,
      };
    })
    .sort(
      (a, b) =>
        a.order - b.order ||
        pageOrder(a.href) - pageOrder(b.href) ||
        a.title.localeCompare(b.title),
    );

  // Keep previous/next links in the same depth-first order as the sidebar.
  const pages = (item: DocNavigationItem): DocPage[] => [
    ...(item.page ? [item.page] : []),
    ...item.children.flatMap(pages),
  ];
  return getDocsNavigation(docs).flatMap(pages);
}

/** Groups pages by their source folders while retaining index order. */
export function getDocsNavigation(docs: DocPage[]): DocNavigationItem[] {
  const roots = new Map<string, DocNavigationItem>();
  const folders = new Map<string, DocNavigationItem>();

  for (const page of docs) {
    const parts = page.entry.id.split("/");
    const filename = parts.pop();

    // Folder identity must not depend on display labels: two sections can
    // have the same title and still own different pages and nested folders.
    const section = parts[0] ?? "";
    const root: DocNavigationItem = roots.get(section) ?? {
      title: page.group,
      children: [],
    };
    roots.set(section, root);
    let parent = root;

    // A folder takes the position of its first indexed page. Its README owns
    // the folder link, so it is not also added as a duplicate child page.
    for (let depth = 1; depth < parts.length; depth++) {
      const path = parts.slice(0, depth + 1).join("/");
      let folder = folders.get(path);
      if (!folder) {
        folder = {
          title: folderTitle(parts[depth]),
          children: [],
        };
        folders.set(path, folder);
        parent.children.push(folder);
      }
      parent = folder;
    }

    if (filename === "README" && parts.length > 1) {
      parent.title = page.title;
      parent.page = page;
      continue;
    }

    const item = { title: page.title, page, children: [] };

    // Top-level sections remain headings; their overview is the first page.
    if (filename === "README") parent.children.unshift(item);
    else parent.children.push(item);
  }

  return [...roots.values()];
}

/** Returns the section and folder ancestors followed by the current page. */
export function getDocTrail(
  items: DocNavigationItem[],
  href: string,
): DocNavigationItem[] {
  for (const item of items) {
    if (item.page?.href === href) return [item];
    const trail = getDocTrail(item.children, href);
    if (trail.length) return [item, ...trail];
  }

  return [];
}

function folderTitle(name: string): string {
  return name
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
