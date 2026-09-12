import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import type { PythonBackendSpec } from "../src/backend-spec.ts";
import { createAgentSession } from "../src/create-agent-session.ts";
import type { BridgeEvent } from "../src/protocol.ts";
import { deferred } from "../src/session-deferred.ts";
import {
  type ApprovalRequest,
  RunError,
  type SessionClose,
  SessionClosedError,
  type SessionDiagnostic,
  type TextChunk,
} from "../src/session-types.ts";

const backend: PythonBackendSpec = {
  app: "packages.process_bridge.tests.subprocess_backend:create_backend",
  projectDir: resolve(import.meta.dir, "../../.."),
};

describe("native events across a real Python subprocess", () => {
  test("tool approval, resolution, continuation, and completion retain source data", async () => {
    const events: BridgeEvent[] = [];
    const text: string[] = [];
    const approved: string[] = [];
    const session = await createAgentSession({
      backend,
      onEvent: (event: BridgeEvent) => events.push(event),
      onAssistantText: (chunk: TextChunk) => text.push(chunk.delta),
      approvals: ({ request }: ApprovalRequest) => {
        approved.push(request.tool_name);
        return true;
      },
    });
    try {
      const result = await session.run("go");
      expect(result).toEqual({
        status: "completed",
        finalOutput: {
          allowed: true,
          text: "界🧪\n".repeat(2000),
          empty: null,
        },
        turnCount: 2,
        responseCount: 0,
      });
      expect(approved).toEqual(["write"]);
      expect(text.join("")).toBe("continued");
      const native = events.flatMap((event) =>
        event.type === "run_event" ? [event.event] : [],
      );
      expect(native.map((event) => event.type)).toEqual([
        "agent_start",
        "model_stream",
        "model_stream",
        "model_stream",
        "model_stream",
        "tool_start",
        "tool_approval",
        "tool_approval",
        "tool_end",
        "model_stream",
        "agent_end",
      ]);
      expect(native[1]).toMatchObject({
        payload: { event: { kind: "text_delta", text: "" } },
      });
      expect(native[2]).toMatchObject({
        payload: { event: { raw: { provider_extra: { unicode: "界🧪\n" } } } },
      });
      expect(native[3]).toMatchObject({
        payload: { event: { kind: "completed" } },
      });
      expect(native[4]).toMatchObject({
        payload: {
          event: {
            kind: "error",
            error: { message: "recoverable model attempt" },
          },
        },
      });
      expect(native[6]).toMatchObject({
        payload: {
          event: {
            record: { status: "pending", request: { tool_call_id: "call-1" } },
          },
        },
      });
      expect(native[7]).toMatchObject({
        payload: {
          event: {
            record: {
              status: "resolved",
              final_decision: { outcome: "allow" },
            },
          },
        },
      });
      expect(
        events.filter((event) => event.type === "run_complete"),
      ).toHaveLength(1);
      expect(events.filter((event) => event.type === "error")).toEqual([]);
      await session.reset();
      expect(events.at(-1)?.type).toBe("reset");
      await expect(session.run("again")).resolves.toMatchObject({
        status: "completed",
      });
    } finally {
      await session.close();
    }
    expect(events.at(-1)?.type).toBe("shutdown");
  }, 20_000);

  test.each([
    "bad-native",
    "bad-result",
    "late-error",
  ])("%s produces one failure and permits the next run", async (prompt) => {
    const events: BridgeEvent[] = [];
    const diagnosticKinds: string[] = [];
    const session = await createAgentSession({
      backend,
      onEvent: (event: BridgeEvent) => events.push(event),
      onDiagnostic: (diagnostic: SessionDiagnostic) =>
        diagnosticKinds.push(diagnostic.kind),
      approvals: () => true,
    });
    try {
      await expect(session.run(prompt)).rejects.toBeInstanceOf(RunError);
      expect(events.filter((event) => event.type === "error")).toHaveLength(1);
      expect(events.filter((event) => event.type === "run_complete")).toEqual(
        [],
      );
      if (prompt === "late-error" || prompt === "bad-result") {
        expect(
          events.some(
            (event) =>
              event.type === "run_event" && event.event.type === "agent_end",
          ),
        ).toBe(true);
      }
      await expect(session.run("recover")).resolves.toMatchObject({
        status: "completed",
      });
    } finally {
      await session.close();
    }
    expect(diagnosticKinds.filter((kind) => kind !== "stderr")).toEqual([]);
  }, 20_000);

  test("cancel closes an active stream and reset permits another run", async () => {
    const waiting = deferred<void>();
    const events: BridgeEvent[] = [];
    const session = await createAgentSession({
      backend,
      approvals: () => true,
      onEvent: (event: BridgeEvent) => events.push(event),
      onAssistantText: (chunk: TextChunk) => {
        if (chunk.text === "waiting") waiting.resolve();
      },
    });
    try {
      const run = session.run("wait");
      await waiting.promise;
      await session.cancel();
      await expect(run).resolves.toEqual({ status: "cancelled" });
      expect(
        events.filter((event) => event.type === "run_cancelled"),
      ).toHaveLength(1);
      await session.reset();
      await expect(session.run("again")).resolves.toMatchObject({
        status: "completed",
      });
    } finally {
      await session.close();
    }
  }, 20_000);

  test.each([
    "reset",
    "close",
  ] as const)("%s releases a pending approval", async (operation) => {
    const pending = deferred<AbortSignal>();
    const events: BridgeEvent[] = [];
    const session = await createAgentSession({
      backend,
      onEvent: (event: BridgeEvent) => events.push(event),
      approvals: ({ signal }: ApprovalRequest) => {
        pending.resolve(signal);
        return new Promise<boolean>(() => {});
      },
    });
    try {
      const run = session.run("go");
      const signal = await pending.promise;
      await session[operation]();
      await expect(run).resolves.toEqual({ status: "cancelled" });
      expect(signal.aborted).toBe(true);
      expect(events.filter((event) => event.type === "run_complete")).toEqual(
        [],
      );
      expect(
        events.filter((event) => event.type === "run_cancelled"),
      ).toHaveLength(1);
    } finally {
      await session.close();
    }
  }, 20_000);

  test("a dead backend rejects the active run without successful completion", async () => {
    const events: BridgeEvent[] = [];
    const closed = deferred<SessionClose>();
    const session = await createAgentSession({
      backend,
      onEvent: (event: BridgeEvent) => events.push(event),
      onClose: (close: SessionClose) => closed.resolve(close),
    });
    try {
      await expect(session.run("disconnect")).rejects.toBeInstanceOf(
        SessionClosedError,
      );
      await expect(closed.promise).resolves.toMatchObject({
        reason: "exit",
        code: 7,
      });
      expect(events.filter((event) => event.type === "run_complete")).toEqual(
        [],
      );
    } finally {
      await session.close();
    }
  }, 20_000);
});
