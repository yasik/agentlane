import { resolve } from "node:path";
import {
  createBridgeChannel,
  spawnBridgeProcess,
  type BridgeChannel,
} from "../../../packages/process_bridge_ts/src/index.ts";

const repoRoot = resolve(import.meta.dir, "../../..");
const backendPath = resolve(import.meta.dir, "backend.py");

let channel: BridgeChannel | null = null;

const child = spawnBridgeProcess(
  {
    command: "uv",
    args: ["run", "python", backendPath],
    cwd: repoRoot,
  },
  {
    onEvent: (event) => {
      console.log(event.type);
      if (event.type === "ready") {
        channel?.send({ type: "prompt", text: "hello from TypeScript" });
      }
      if (event.type === "run_complete") {
        channel?.shutdown();
      }
    },
    onInvalidLine: (line) => console.error(`invalid stdout: ${line}`),
    onDecodeError: (error) => console.error(`decode error: ${error.message}`),
    onStderr: (line) => console.error(line),
  },
);

channel = createBridgeChannel(child, {
  onFinalize: () => undefined,
});
