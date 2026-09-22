import type { APIRoute } from "astro";

/** Publish crawler rules for the same origin used by canonical links. */
export const GET: APIRoute = ({ site }) =>
  new Response(
    import.meta.env.VERCEL_ENV === "preview"
      ? "User-agent: *\nDisallow: /\n"
      : `User-agent: *\nAllow: /\n\nSitemap: ${new URL("sitemap-index.xml", site).href}\n`,
    { headers: { "Content-Type": "text/plain; charset=utf-8" } },
  );
