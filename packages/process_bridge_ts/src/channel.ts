import { type BridgeCommand, encodeBridgeCommand } from "./protocol.ts";

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

export type BridgeChannel = {
  send: (command: BridgeCommand) => boolean;
  shutdown: () => void;
  isShuttingDown: () => boolean;
};

export type ChannelScheduler = {
  set: (fn: () => void, ms: number) => unknown;
  clear: (handle: unknown) => void;
};

export type BridgeChannelOptions = {
  graceMs?: number;
  scheduler?: ChannelScheduler;
  onSendError?: (message: string) => void;
  onFinalize?: () => void;
};

const SHUTDOWN_GRACE_MS = 1500;

const REAL_SCHEDULER: ChannelScheduler = {
  set: (fn: () => void, ms: number): unknown => setTimeout(fn, ms),
  clear: (handle: unknown): void =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};

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
    if (!sendRaw({ type: "shutdown" })) child.kill();

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
