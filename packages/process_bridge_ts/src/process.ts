import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import type { Readable } from "node:stream";
import { BridgeDecodeError, decodeBridgeEventLine } from "./decoders.ts";
import type { BridgeEvent } from "./protocol.ts";

/** Callbacks used by process wiring to report protocol, stderr, and lifecycle signals. */
export type BridgeProcessCallbacks = {
  /** Called for malformed stdout protocol frames before they are dropped. */
  onDecodeError?: (error: BridgeDecodeError, line: string) => void;

  /** Called for each strict, successfully decoded backend event. */
  onEvent: (event: BridgeEvent) => void;

  /** Called after the child closes and readline resources are disposed. */
  onExit?: (code: number | null, signal: NodeJS.Signals | null) => void;

  /** Called for stdout lines that are not valid bridge events. */
  onInvalidLine?: (line: string) => void;

  /** Called if Node fails to spawn or manage the child process. */
  onSpawnError?: (error: Error) => void;

  /** Called for each backend stderr diagnostic line. */
  onStderr?: (line: string) => void;
};

/** Options used to spawn the Python bridge backend process. */
export type BridgeProcessOptions = {
  /** Executable name or path, for example `uv`. */
  command: string;

  /** Arguments passed to `command`. */
  args?: string[];

  /** Working directory for the child process. */
  cwd?: string;

  /** Environment variables for the child process. Defaults to `process.env`. */
  env?: NodeJS.ProcessEnv;

  /** Whether to spawn through a shell. Defaults to false. */
  shell?: boolean;
};

/** Readable side of a bridge process or process-like test double. */
export type BridgeReadableProcess = {
  stdout: Readable;
  stderr: Readable;
};

/** Disposable wiring created around stdout and stderr readline streams. */
export type BridgeProcessWiring = {
  /** Close the readline wrappers. Safe to call more than once. */
  dispose: () => void;
};

/**
 * Spawn a backend process and wire its stdout/stderr streams.
 *
 * stdout is treated as the NDJSON protocol channel; stderr is forwarded as
 * diagnostics. The returned child still exposes stdin so callers can create a
 * `BridgeChannel` for app-to-backend commands.
 */
export function spawnBridgeProcess(
  options: BridgeProcessOptions,
  callbacks: BridgeProcessCallbacks,
): ChildProcessWithoutNullStreams {
  const child = spawn(options.command, options.args ?? [], {
    cwd: options.cwd,
    env: options.env ?? process.env,
    shell: options.shell ?? false,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const wiring = wireBridgeProcess(child, callbacks);

  child.on("error", (error: Error): void => {
    // Spawn errors are process-management failures, not protocol frames.
    callbacks.onSpawnError?.(error);
  });

  child.on(
    "close",
    (code: number | null, signal: NodeJS.Signals | null): void => {
      // Dispose before reporting exit so host cleanup does not race lingering
      // readline handles.
      wiring.dispose();
      callbacks.onExit?.(code, signal);
    },
  );

  return child;
}

/**
 * Wire an existing process-like object's stdout and stderr streams.
 *
 * Tests use this with in-memory streams, while production uses the child
 * returned by `spawnBridgeProcess`. Keeping wiring separate makes stream
 * behavior testable without creating real processes.
 */
export function wireBridgeProcess(
  child: BridgeReadableProcess,
  callbacks: BridgeProcessCallbacks,
): BridgeProcessWiring {
  // stdout is the protocol channel. Backend diagnostics must use stderr so
  // malformed logs never masquerade as bridge events.
  const stdout = createInterface({ input: child.stdout });

  stdout.on("line", (line: string): void => {
    let event: BridgeEvent;
    try {
      event = decodeBridgeEventLine(line);
    } catch (error) {
      const decodeError =
        error instanceof BridgeDecodeError
          ? error
          : new BridgeDecodeError(`Unexpected bridge decode failure: ${error}`);
      // Invalid stdout is both a protocol diagnostic and an invalid-line signal
      // for hosts that still want to keep a raw trace of dropped frames.
      callbacks.onDecodeError?.(decodeError, line);
      callbacks.onInvalidLine?.(line);
      return;
    }

    callbacks.onEvent(event);
  });

  const stderr = createInterface({ input: child.stderr });
  stderr.on("line", (line: string): void => {
    callbacks.onStderr?.(line);
  });

  let disposed = false;

  return {
    dispose: () => {
      if (disposed) return;

      // The child close callback can race with final stdout/stderr delivery;
      // disposal must be idempotent and only close the readline wrappers.
      disposed = true;

      stdout.close();
      stderr.close();
    },
  };
}
