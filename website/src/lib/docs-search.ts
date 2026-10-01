export type SearchPage = {
  title: string;
  description: string;
  href: string;
  group: string;
  text: string;
};

type SearchEntry = {
  page: SearchPage;
  title: string;
  body: string;
};

/** Normalizes document text once for repeated searches in the browser. */
export function prepareSearchIndex(pages: SearchPage[]): SearchEntry[] {
  return pages.map((page) => ({
    page,
    title: page.title.toLowerCase(),
    body: `${page.title} ${page.text}`.toLowerCase(),
  }));
}

/** Matches every search term and ranks page titles before body text. */
export function searchDocs(index: SearchEntry[], query: string): SearchPage[] {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  if (!terms.length) return [];
  return index
    .map(({ page, title, body }) => {
      return {
        page,
        score: terms.every((term) => body.includes(term))
          ? terms.reduce(
              (score, term) => score + (title.includes(term) ? 10 : 1),
              0,
            )
          : 0,
      };
    })
    .filter(({ score }) => score > 0)
    .sort(
      (a, b) => b.score - a.score || a.page.title.localeCompare(b.page.title),
    )
    .slice(0, 12)
    .map(({ page }) => page);
}
