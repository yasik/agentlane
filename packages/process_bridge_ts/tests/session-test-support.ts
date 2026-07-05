import type { BridgeChildLike } from "../src/channel.ts";
import type { BridgeReadableProcess } from "../src/process.ts";
import { wireBridgeProcess } from "../src/process.ts";
import { AgentSessionController } from "../src/session.ts";
import type {
  AgentSession,
  AgentSessionOptions,
} from "../src/session-types.ts";

type SessionChild = BridgeChildLike &
  BridgeReadableProcess & {
    on: {
      (
        event: "close",
        listener: (code: number | null, signal: NodeJS.Signals | null) => void,
      ): unknown;
      (event: "error", listener: (error: Error) => void): unknown;
    };
  };

/**
 * Attach the session layer to a process-like test double.
 *
 * This mirrors `spawnBridgeProcess` wiring without creating a real child
 * process, so tests can drive exact stdout, stderr, close, and write-failure
 * sequences without weakening the production process abstraction.
 */
export function attachAgentSession<
  TConfig extends Record<string, unknown> = Record<string, unknown>,
>(
  child: SessionChild,
  options: AgentSessionOptions<TConfig>,
): Promise<AgentSession<TConfig>> {
  const controller = new AgentSessionController<TConfig>(options);
  const wiring = wireBridgeProcess(child, controller.processCallbacks());
  child.on("error", (error: Error): void => {
    controller.handleSpawnError(error);
  });
  child.on(
    "close",
    (code: number | null, signal: NodeJS.Signals | null): void => {
      wiring.dispose();
      controller.handleExit(code, signal);
    },
  );
  return controller.start(child);
}
