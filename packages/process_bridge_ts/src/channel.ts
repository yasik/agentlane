import { type BridgeCommand, encodeBridgeCommand } from "./protocol.ts";

/** Minimal child-process shape needed to send commands and own shutdown. */
export type BridgeChildLike = {
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  kill: (signal?: NodeJS.Signals) => boolean;
  once: (event: "exit", listener: () => void) => unknown;
  stdin: {
    destroyed: boolean;
    writable: boolean;
    write: (chunk: string) => boolean;
    on: (event: "error", listener: (error: Error) => void) => unknown;
  };
};

/** Writable command channel from the app into the Python bridge backend. */
export type BridgeChannel = {
  /** Send one command frame. Returns false after shutdown or stdin failure. */
  send: (command: BridgeCommand) => boolean;

  /** Request cooperative backend shutdown, then escalate after the grace window. */
  shutdown: () => void;

  /** Report whether the channel has stopped accepting new commands. */
  isShuttingDown: () => boolean;
};

/** Timer adapter used by tests to make shutdown escalation deterministic. */
export type ChannelScheduler = {
  /** Schedule a callback and return an opaque timer handle. */
  set: (fn: () => void, ms: number) => unknown;

  /** Clear a previously scheduled timer handle. */
  clear: (handle: unknown) => void;
};

/** Options for command-channel error reporting and shutdown timing. */
export type BridgeChannelOptions = {
  /** Milliseconds to wait for cooperative backend exit before SIGKILL. */
  graceMs?: number;

  /** Optional scheduler override for deterministic tests. */
  scheduler?: ChannelScheduler;

  /** Called when a command cannot be written before shutdown begins. */
  onSendError?: (message: string) => void;

  /** Called once after process exit or forced shutdown cleanup completes. */
  onFinalize?: () => void;
};

/** Default grace period for a backend to observe the shutdown command. */
const SHUTDOWN_GRACE_MS = 1500;

/** Production scheduler backed by Node timers. */
const REAL_SCHEDULER: ChannelScheduler = {
  set: (fn: () => void, ms: number): unknown => setTimeout(fn, ms),
  clear: (handle: unknown): void =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};

/**
 * Create the stdin command channel for a spawned bridge backend.
 *
 * The channel treats stdin as the only command transport. Shutdown first sends a
 * protocol `shutdown` command so Python can flush final events, then escalates
 * if the process does not exit inside the configured grace period.
 */
export function createBridgeChannel(
  child: BridgeChildLike,
  options: BridgeChannelOptions = {},
): BridgeChannel {
  const graceMs = options.graceMs ?? SHUTDOWN_GRACE_MS;
  const scheduler = options.scheduler ?? REAL_SCHEDULER;
  let shuttingDown = false;
  let finalized = false;
  const timers: unknown[] = [];

  const reportSendError = (message: string): void => {
    if (!shuttingDown) options.onSendError?.(message);
  };

  child.stdin.on("error", (error: Error): void => {
    // Stream errors can arrive outside a direct send call. Once shutdown has
    // started they are expected noise, so only report pre-shutdown failures.
    reportSendError(error.message);
  });

  const sendRaw = (command: BridgeCommand): boolean => {
    if (child.stdin.destroyed || !child.stdin.writable) {
      reportSendError(`bridge stdin is closed (command: ${command.type})`);
      return false;
    }

    try {
      // Writable.write(false) still accepted and queued the bytes; it only
      // signals backpressure. Treat thrown writes and closed streams as failure.
      child.stdin.write(encodeBridgeCommand(command));
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      reportSendError(`bridge stdin write failed: ${message}`);
      return false;
    }
  };

  const finalize = (): void => {
    if (finalized) return;

    // Process exit and grace timers can fire in either order. Finalization owns
    // timer cleanup and must call the host callback at most once.
    finalized = true;

    timers.forEach((timer: unknown): void => {
      scheduler.clear(timer);
    });

    options.onFinalize?.();
  };

  const shutdown = (): void => {
    if (shuttingDown) return;

    shuttingDown = true;

    if (child.exitCode !== null || child.signalCode !== null) {
      finalize();
      return;
    }

    child.once("exit", finalize);

    // Try cooperative shutdown first. If stdin is already unavailable, the
    // backend cannot observe the command, so immediate termination is correct.
    if (!sendRaw({ type: "shutdown" })) {
      child.kill();
    }

    timers.push(scheduler.set(() => child.kill("SIGKILL"), graceMs));
    timers.push(scheduler.set(finalize, graceMs + 250));
  };

  return {
    send: (command: BridgeCommand): boolean =>
      shuttingDown ? false : sendRaw(command),
    shutdown,
    isShuttingDown: (): boolean => shuttingDown,
  };
}
