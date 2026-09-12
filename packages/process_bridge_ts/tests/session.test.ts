import { describe, expect, test } from "bun:test";
import {
  RunError,
  type SessionClose,
  SessionClosedError,
  type SessionDiagnostic,
  type TextChunk,
  type ToolActivity,
} from "../src/session-types.ts";
import { FakeChild } from "./session-test-helpers.ts";
import { attachAgentSession } from "./session-test-support.ts";

describe("agent session", () => {
  test("rejects startup when ready never arrives", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      readyTimeoutMs: 1,
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });

    await expect(sessionPromise).rejects.toThrow("Timed out waiting");
    expect(child.killed).toContain("SIGKILL");
    expect(closes).toEqual([]);
  });

  test("rejects startup when the backend exits before ready", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });

    child.emitClose(1, null);

    await expect(sessionPromise).rejects.toThrow("Backend exited before ready");
    expect(closes).toEqual([]);
  });

  test("runs a prompt and delivers typed text", async () => {
    const child = new FakeChild();
    const text: TextChunk[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      textDelivery: "immediate",
      onAssistantText: (chunk: TextChunk): void => {
        text.push(chunk);
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    expect(child.commands().at(-1)).toMatchObject({
      type: "prompt",
      text: "hello",
    });
    child.emitEvent({ type: "run_start", ts: 2, prompt: "hello" });
    child.emitModel("text_delta", { text: "Echo: hello" });
    child.emitEvent({
      type: "run_complete",
      ts: 4,
      final_output: "Echo: hello",
      turn_count: 1,
      response_count: 0,
      shim_state: {},
    });

    await expect(run).resolves.toEqual({
      status: "completed",
      finalOutput: "Echo: hello",
      turnCount: 1,
      responseCount: 0,
    });
    expect(text.at(-1)).toMatchObject({ text: "Echo: hello", done: true });
  });

  test("attributes command errors to the pending run", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    child.emitEvent({
      type: "error",
      ts: 2,
      message: "A run is already active.",
      scope: "command",
    });

    await expect(run).rejects.toBeInstanceOf(RunError);
  });

  test("sweeps open tool calls on cancellation", async () => {
    const child = new FakeChild();
    const tools: ToolActivity[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onToolActivity: (activity: ToolActivity): void => {
        tools.push(activity);
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    child.emitEvent({ type: "run_start", ts: 2, prompt: "hello" });
    child.emitNative("tool_start", {
      task_id: "task",
      parent_task_id: null,
      is_root: true,
      task_name: "Agent",
      tool_call: {
        id: "call_1",
        type: "function",
        function: { name: "read", arguments: "{}" },
      },
      is_delegation: false,
    });
    child.emitEvent({ type: "run_cancelled", ts: 4 });

    await expect(run).resolves.toEqual({ status: "cancelled" });
    expect(tools.map((tool) => tool.phase)).toEqual(["start", "cancelled"]);
  });

  test("reset cancels the active run before resolving reset", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    child.emitEvent({ type: "run_start", ts: 2, prompt: "hello" });
    const reset = session.reset();

    expect(child.commands().at(-1)).toMatchObject({ type: "reset" });
    child.emitEvent({ type: "run_cancelled", ts: 3 });
    child.emitEvent({ type: "reset", ts: 4 });

    await expect(run).resolves.toEqual({ status: "cancelled" });
    await expect(reset).resolves.toBeUndefined();
  });

  test("handler errors become diagnostics without breaking run settlement", async () => {
    const child = new FakeChild();
    const diagnostics: string[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      textDelivery: "immediate",
      onAssistantText: (): void => {
        throw new Error("render failed");
      },
      onDiagnostic: (diagnostic: SessionDiagnostic): void => {
        if (diagnostic.kind === "handler-error") {
          diagnostics.push(diagnostic.handler);
        }
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    child.emitEvent({ type: "run_start", ts: 2, prompt: "hello" });
    child.emitModel("text_delta", { text: "Echo: hello" });
    child.emitEvent({
      type: "run_complete",
      ts: 4,
      final_output: "Echo: hello",
      turn_count: 1,
      response_count: 0,
      shim_state: {},
    });

    await expect(run).resolves.toMatchObject({ status: "completed" });
    expect(diagnostics).toContain("onAssistantText");
  });

  test("send failures close the session and reject active work", async () => {
    const child = new FakeChild();
    const diagnostics: string[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onDiagnostic: (diagnostic: SessionDiagnostic): void => {
        diagnostics.push(diagnostic.kind);
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    child.writeError = new Error("disk full");
    const run = session.run("hello");

    await expect(run).rejects.toBeInstanceOf(SessionClosedError);
    expect(diagnostics).toContain("send-failed");
    expect(child.killed).toContain("SIGKILL");
  });

  test("fatal protocol errors close the session and reject active runs", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const text: TextChunk[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onDiagnostic: (_diagnostic: SessionDiagnostic): void => undefined,
      onAssistantText: (chunk: TextChunk): void => {
        text.push(chunk);
      },
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const run = session.run("hello");

    child.emitLine("not-json");
    child.emitModel("text_delta", { text: "stale" });

    await expect(run).rejects.toBeInstanceOf(SessionClosedError);
    expect(closes[0]?.reason).toBe("protocol-error");
    expect(child.killed).toContain("SIGKILL");
    expect(text).toEqual([]);
  });

  test("close sends shutdown and resolves after process close", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });
    child.emitEvent({
      type: "ready",
      ts: 1,
      version: "0.1.0",
      package: "agentlane-process-bridge",
    });
    const session = await sessionPromise;
    const close = session.close();

    expect(child.commands().at(-1)).toMatchObject({ type: "shutdown" });
    child.emitEvent({ type: "shutdown", ts: 2 });
    child.emitClose(0, null);

    await expect(close).resolves.toBeUndefined();
    expect(closes).toEqual([{ reason: "shutdown", code: 0, signal: null }]);
  });
});
