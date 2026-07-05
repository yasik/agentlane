import { resolveBackendSpec } from "./backend-spec.ts";
import type { BridgeProcessOptions } from "./process.ts";
import { spawnBridgeProcess } from "./process.ts";
import { AgentSessionController } from "./session.ts";
import type { AgentSession, AgentSessionOptions } from "./session-types.ts";
import { SessionStartError } from "./session-types.ts";

/**
 * Spawn the backend, wire stdio, and resolve once `ready` arrives.
 *
 * Startup failures reject with `SessionStartError` before a live session is
 * exposed. After startup, all operation promises settle through the session
 * lifecycle; app code never needs to observe the raw child process.
 */
export function createAgentSession<
  TConfig extends object = Record<string, unknown>,
  TConfigPatch extends object = Partial<TConfig>,
>(
  options: AgentSessionOptions<TConfig>,
): Promise<AgentSession<TConfig, TConfigPatch>> {
  let processOptions: BridgeProcessOptions;
  try {
    processOptions = resolveBackendSpec(options.backend);
  } catch (error) {
    return Promise.reject(
      new SessionStartError(
        error instanceof Error ? error.message : String(error),
      ),
    );
  }

  const controller = new AgentSessionController<TConfig, TConfigPatch>(options);
  const child = spawnBridgeProcess(
    processOptions,
    controller.processCallbacks(),
  );
  return controller.start(child);
}
