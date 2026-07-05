import type { AgentSession, ReadyInfo, RunResult } from "./session-types.ts";

/**
 * Controller operations exposed through the public session handle.
 *
 * The handle is intentionally a facade: it has stable methods for app code, but
 * all mutable lifecycle state remains owned by `AgentSessionController`.
 */
type SessionHandleOperations<TConfig extends Record<string, unknown>> = {
  /** Read the latest backend-announced config document. */
  getConfig: () => Readonly<TConfig> | undefined;

  /** Start one prompt run. */
  run: (text: string) => Promise<RunResult>;

  /** Request cancellation of the active run. */
  cancel: () => Promise<void>;

  /** Reset backend conversation state. */
  reset: () => Promise<void>;

  /** Send a runtime config patch and await the authoritative applied document. */
  configure: (patch: Partial<TConfig>) => Promise<Readonly<TConfig>>;

  /** Gracefully close the local backend process. */
  close: () => Promise<void>;
};

/** Build the public handle while keeping mutable session state in the controller. */
export function createSessionHandle<
  TConfig extends Record<string, unknown> = Record<string, unknown>,
>(
  ready: ReadyInfo,
  operations: SessionHandleOperations<TConfig>,
): AgentSession<TConfig> {
  return {
    ready,
    // This getter is load-bearing: a plain value would freeze the startup
    // config and miss later configure/reset acknowledgements.
    get config(): Readonly<TConfig> | undefined {
      return operations.getConfig();
    },
    run: operations.run,
    cancel: operations.cancel,
    reset: operations.reset,
    configure: operations.configure,
    close: operations.close,
  };
}
