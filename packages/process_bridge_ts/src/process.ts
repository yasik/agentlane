import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import type { Readable } from "node:stream";
import { decodeBridgeEventLine } from "./decoders.ts";
import type { BridgeEvent } from "./protocol.ts";

export type BridgeProcessCallbacks = {
  onDecodeFallback?: (eventType: string, fields: string[]) => void;
  onEvent: (event: BridgeEvent) => void;
  onExit?: (code: number | null, signal: NodeJS.Signals | null) => void;
  onInvalidLine?: (line: string) => void;
  onSpawnError?: (error: Error) => void;
  onStderr?: (line: string) => void;
};

export type BridgeProcessOptions = {
  command: string;
  args?: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  shell?: boolean;
};

export type BridgeReadableProcess = {
  stdout: Readable;
  stderr: Readable;
};

export type BridgeProcessWiring = {
  dispose: () => void;
};

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
    callbacks.onSpawnError?.(error);
  });

  child.on(
    "close",
    (code: number | null, signal: NodeJS.Signals | null): void => {
      wiring.dispose();
      callbacks.onExit?.(code, signal);
    },
  );

  return child;
}

export function wireBridgeProcess(
  child: BridgeReadableProcess,
  callbacks: BridgeProcessCallbacks,
): BridgeProcessWiring {
  // stdout is the protocol channel. Backend diagnostics must use stderr so
  // malformed logs never masquerade as bridge events.
  const stdout = createInterface({ input: child.stdout });

  stdout.on("line", (line: string): void => {
    const decoded = decodeBridgeEventLine(line);

    if (decoded === null) {
      callbacks.onInvalidLine?.(line);
      return;
    }

    if (decoded.fallbacks.length > 0) {
      // A fallback means the decoder synthesized required data. Report the
      // drift, but do not feed repaired state into app reducers.
      callbacks.onDecodeFallback?.(decoded.event.type, decoded.fallbacks);
      return;
    }

    callbacks.onEvent(decoded.event);
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
