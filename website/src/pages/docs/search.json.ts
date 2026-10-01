import { getDocs } from "../../lib/docs";

export async function GET(): Promise<Response> {
  const docs = await getDocs();
  return Response.json(
    docs.map(({ title, description, href, group, text }) => ({
      title,
      description,
      href,
      group,
      text,
    })),
  );
}
