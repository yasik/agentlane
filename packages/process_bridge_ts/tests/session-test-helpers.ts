import { PassThrough } from "node:stream";
import type { BridgeChildLike } from "../src/channel.ts";

export class FakeChild implements BridgeChildLike {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  killed: Array<NodeJS.Signals | undefined> = [];
  writes: string[] = [];
  private closeListeners: Array<
    (code: number | null, signal: NodeJS.Signals | null) => void
  > = [];
  private exitOnce: (() => void) | null = null;
  stdin = {
    destroyed: false,
    writable: true,
    write: (chunk: string): boolean => {
      this.writes.push(chunk);
      return true;
    },
    on: (_event: "error", _listener: (error: Error) => void): unknown =>
      undefined,
  };

  kill(signal?: NodeJS.Signals): boolean {
    this.killed.push(signal);
    return true;
  }

  once(_event: "exit", listener: () => void): unknown {
    this.exitOnce = listener;
    return undefined;
  }

  on(
    event: "close" | "error",
    listener:
      | ((code: number | null, signal: NodeJS.Signals | null) => void)
      | ((error: Error) => void),
  ): unknown {
    if (event === "close") {
      this.closeListeners.push(
        listener as (
          code: number | null,
          signal: NodeJS.Signals | null,
        ) => void,
      );
    }
    return undefined;
  }

  emitReady(): void {
    this.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
  }

  emitApprovalRequest(id: string): void {
    this.emitEvent({
      type: "approval_request",
      ts: 2,
      id,
      request: approvalRequestPayload(),
      reason: "review",
    });
  }

  emitApprovalResolved(id: string, allowed: boolean): void {
    this.emitEvent({
      type: "approval_resolved",
      ts: 3,
      id,
      allowed,
      request: approvalRequestPayload(),
      reason: null,
    });
  }

  emitEvent(event: Record<string, unknown>): void {
    this.stdout.write(
      `${JSON.stringify({ protocol_version: "1.0", ...event })}\n`,
    );
  }

  emitClose(
    code: number | null = 0,
    signal: NodeJS.Signals | null = null,
  ): void {
    this.exitCode = code;
    this.signalCode = signal;
    this.exitOnce?.();
    for (const listener of this.closeListeners) listener(code, signal);
  }

  commands(): Array<Record<string, unknown>> {
    return this.writes.map(
      (line: string): Record<string, unknown> => JSON.parse(line),
    );
  }
}

function approvalRequestPayload(): Record<string, unknown> {
  return {
    tool_name: "write",
    operation: "create_file",
    cwd: "/workspace",
    path: "/workspace/a.txt",
    command: null,
    skill_name: null,
    reason: null,
    run_id: null,
    agent_name: null,
    tool_call_id: "call_1",
    metadata: {},
  };
}
