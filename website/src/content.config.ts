import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const copy = z.object({
  id: z.string(),
  status: z.string(),
  active: z.boolean(),
});
const role = z.object({
  id: z.string(),
  name: z.string(),
  duty: z.string(),
  status: z.string(),
  annotation: z.string().optional(),
  copies: z.boolean().optional(),
  coverage: z.boolean().optional(),
  independent: z.boolean().optional(),
  get children() {
    return z.array(role).optional();
  },
});

const examples = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/examples" }),
  schema: z
    .object({
      id: z.string(),
      order: z.number().int().nonnegative(),
      label: z.string(),
      filename: z.string(),
      note: z.string(),
      duration: z.number().positive(),
      channels: z.array(z.object({ id: z.string(), label: z.string() })),
      roles: z.array(role),
      scenes: z
        .array(
          z.object({
            at: z.number().nonnegative(),
            route: z.string(),
            text: z.string(),
            active: z.array(z.string()),
            channel: z.string(),
            states: z.record(z.string(), z.string()),
            copies: z.array(copy),
            copyLabel: z.string(),
            coverage: z.array(copy).optional(),
            gate: z.string().optional(),
            context: z.string(),
            ready: z.boolean().optional(),
          }),
        )
        .min(1),
      draft: z
        .object({
          title: z.string(),
          note: z.string(),
          sections: z.array(z.object({ title: z.string(), text: z.string() })),
        })
        .optional(),
    })
    .refine(
      (example) =>
        example.scenes[0].at === 0 &&
        example.scenes.every(
          (scene, index) =>
            scene.at < (example.scenes[index + 1]?.at ?? example.duration),
        ),
      {
        message:
          "Scenes must start at zero, increase in time, and finish before duration.",
      },
    ),
});

const pages = defineCollection({
  loader: glob({ pattern: "*.{md,mdx}", base: "./src/content/pages" }),
  schema: z.object({
    title: z.string().min(1),
    description: z.string().min(1),
    socialImage: z.string().default("/og.png"),
    socialImageAlt: z
      .string()
      .default("AgentLane — A runtime for autonomous organizations"),
    noindex: z.boolean().default(false),
  }),
});

const docs = defineCollection({
  loader: glob({
    base: "../docs",
    pattern: ["**/*.md", "!plans/**"],
    generateId: ({ entry }) => entry.replace(/\.md$/, ""),
  }),
});

export const collections = { examples, pages, docs };
