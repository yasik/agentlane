import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";
import { loadEnv } from "vite";

const env = loadEnv(process.env.NODE_ENV ?? "production", process.cwd(), "");
const origin = new URL(env.SITE_URL || "https://getagentlane.dev");
if (
  !["http:", "https:"].includes(origin.protocol) ||
  origin.pathname !== "/" ||
  origin.search ||
  origin.hash
) {
  throw new Error(
    "SITE_URL must be an HTTP(S) origin without a path, query, or fragment.",
  );
}

export default defineConfig({
  site: origin.origin,
  output: "static",
  integrations: [mdx(), react(), sitemap()],
  markdown: {
    shikiConfig: {
      themes: { dark: "github-dark", light: "github-light-high-contrast" },
      defaultColor: false,
    },
  },
  vite: { build: { sourcemap: false } },
});
