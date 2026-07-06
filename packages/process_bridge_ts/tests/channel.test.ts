import { describe, expect, test } from "bun:test";
import { type BridgeChildLike, createBridgeChannel } from "../src/channel.ts";

class FakeChild implements BridgeChildLike {
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  errorListener: ((error: Error) => void) | null = null;
  killed: (NodeJS.Signals | undefined)[] = [];
  writes: string[] = [];
  writeError: Error | null = null;
  writeResult = true;
  exitListener: (() => void) | null = null;
  stdin = {
    destroyed: false,
    writable: true,
    write: (chunk: string): boolean => {
      if (this.writeError !== null) throw this.writeError;
      this.writes.push(chunk);
      return this.writeResult;
    },
    on: (_event: "error", listener: (error: Error) => void): unknown => {
      this.errorListener = listener;
      return undefined;
    },
  };

  kill(signal?: NodeJS.Signals): boolean {
    this.killed.push(signal);
    return true;
  }

  once(_event: "exit", listener: () => void): unknown {
    this.exitListener = listener;
    return undefined;
  }

  emitStdinError(error: Error): void {
    this.errorListener?.(error);
  }
}

describe("bridge channel", () => {
  test("sends encoded commands until shutdown", () => {
    const child = new FakeChild();
    const channel = createBridgeChannel(child);

    expect(channel.send({ type: "prompt", text: "hello" })).toBe(true);
    expect(JSON.parse(child.writes[0] ?? "{}")).toMatchObject({
      protocol_version: "1.0",
      type: "prompt",
      text: "hello",
    });
  });

  test("shutdown sends polite command, escalates, and finalizes once", () => {
    const child = new FakeChild();
    const callbacks: (() => void)[] = [];
    let finalized = 0;
    const channel = createBridgeChannel(child, {
      graceMs: 10,
      scheduler: {
        set: (fn: () => void): unknown => {
          callbacks.push(fn);
          return fn;
        },
        clear: (_handle: unknown): void => undefined,
      },
      onFinalize: () => {
        finalized += 1;
      },
    });

    channel.shutdown();
    expect(channel.isShuttingDown()).toBe(true);
    expect(channel.send({ type: "cancel" })).toBe(false);
    expect(JSON.parse(child.writes[0] ?? "{}")).toMatchObject({
      type: "shutdown",
    });

    callbacks[0]?.();
    callbacks[1]?.();
    callbacks[1]?.();

    expect(child.killed).toEqual(["SIGKILL"]);
    expect(finalized).toBe(1);
  });

  test("reports stdin send failures without accepting the command", () => {
    const child = new FakeChild();
    const errors: string[] = [];
    const channel = createBridgeChannel(child, {
      onSendError: (message: string): void => {
        errors.push(message);
      },
    });

    child.stdin.writable = false;
    expect(channel.send({ type: "prompt", text: "closed" })).toBe(false);
    child.stdin.writable = true;
    child.writeError = new Error("boom");
    expect(channel.send({ type: "prompt", text: "throws" })).toBe(false);
    child.writeError = null;
    child.emitStdinError(new Error("async boom"));

    expect(errors).toEqual([
      "bridge stdin is closed (command: prompt)",
      "bridge stdin write failed: boom",
      "async boom",
    ]);
  });

  test("accepts commands queued under stdin backpressure", () => {
    const child = new FakeChild();
    const errors: string[] = [];
    const channel = createBridgeChannel(child, {
      onSendError: (message: string): void => {
        errors.push(message);
      },
    });

    child.writeResult = false;

    expect(channel.send({ type: "prompt", text: "queued" })).toBe(true);
    expect(errors).toEqual([]);
    expect(JSON.parse(child.writes[0] ?? "{}")).toMatchObject({
      type: "prompt",
      text: "queued",
    });
  });

  test("shutdown does not kill immediately when polite shutdown is queued", () => {
    const child = new FakeChild();
    const callbacks: (() => void)[] = [];
    const channel = createBridgeChannel(child, {
      graceMs: 10,
      scheduler: {
        set: (fn: () => void): unknown => {
          callbacks.push(fn);
          return fn;
        },
        clear: (_handle: unknown): void => undefined,
      },
    });

    child.writeResult = false;

    channel.shutdown();

    expect(child.killed).toEqual([]);
    callbacks[0]?.();
    expect(child.killed).toEqual(["SIGKILL"]);
  });

  test("shutdown kills immediately when stdin is already closed", () => {
    const child = new FakeChild();
    const channel = createBridgeChannel(child);

    child.stdin.writable = false;
    channel.shutdown();

    expect(child.killed).toEqual([undefined]);
  });
});
