import { resolve } from "node:path";
import { createAgentSession, RunError } from "@agentlane/process-bridge";

const repoRoot = resolve(import.meta.dir, "../../..");

const session = await createAgentSession({
  backend: {
    app: "examples.harness.process_bridge_stdio.backend:create_backend",
    projectDir: repoRoot,
  },
  onEvent: (event) => {
    console.log(event.type);
  },
  onAssistantText: (chunk) => {
    if (chunk.done) {
      console.log(`assistant: ${chunk.text}`);
    }
  },
});

try {
  const result = await session.run("hello from TypeScript");

  if (result.status === "cancelled") {
    console.log("run_cancelled");
  }
} catch (error) {
  if (error instanceof RunError) {
    console.error(`run failed: ${error.message}`);
  } else {
    throw error;
  }
} finally {
  await session.close();
}
