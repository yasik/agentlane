import { getCollection, type CollectionEntry } from "astro:content";
import { z } from "astro/zod";
import { docsRoute } from "./docs-paths.mjs";

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
  return entries
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
        group: groups[group] ?? group.replaceAll("-", " "),
        order: groupIndex < 0 ? Object.keys(groups).length : groupIndex,
      };
    })
    .sort(
      (a, b) =>
        a.order - b.order ||
        Number(b.entry.id.endsWith("README")) -
          Number(a.entry.id.endsWith("README")) ||
        pageOrder(a.href) - pageOrder(b.href) ||
        a.title.localeCompare(b.title),
    );
}
