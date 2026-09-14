import { describe, expect, test } from "bun:test";
import type { BridgeEvent } from "../src/protocol.ts";
import type {
  AgentActivity,
  TextChunk,
  ToolActivity,
} from "../src/session-types.ts";
import { FakeChild } from "./session-test-helpers.ts";
import { attachAgentSession } from "./session-test-support.ts";

const resultValues: unknown[] = [
  null,
  false,
  42,
  [1, true, null],
  {
    items: Array.from({ length: 60 }, (_, index) => ({
      index,
      text: "界🧪".repeat(6000),
    })),
  },
];

describe("session payload fidelity", () => {
  test.each(
    resultValues,
  )("delivers structured tool, agent, and run results: %#", async (value: unknown): Promise<void> => {
    const child = new FakeChild();
    const text: TextChunk[] = [];
    const tools: ToolActivity[] = [];
    const agents: AgentActivity[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onAssistantText: (chunk: TextChunk): void => {
        text.push(chunk);
      },
      onToolActivity: (activity: ToolActivity): void => {
        tools.push(activity);
      },
      onAgentActivity: (activity: AgentActivity): void => {
        agents.push(activity);
      },
    });
    child.emitReady();
    const session = await sessionPromise;
    const run = session.run("go");
    child.emitEvent({ type: "run_start", ts: 2, prompt: "go" });
    const lineage = {
      task_id: "root",
      parent_task_id: null,
      is_root: true,
      task_name: "Root",
    };
    child.emitNative("agent_start", lineage);
    const tool_call = {
      id: "call",
      type: "function",
      function: { name: "read", arguments: JSON.stringify({ value }) },
    };
    child.emitNative("tool_start", {
      ...lineage,
      tool_call,
      is_delegation: false,
    });
    child.emitNative("tool_end", {
      ...lineage,
      tool_call,
      result: value,
      ok: true,
      error: null,
      is_delegation: false,
    });
    child.emitNative("agent_end", {
      ...lineage,
      result: {
        final_output: value,
        responses: [],
        turn_count: 1,
        run_state: null,
      },
    });
    child.emitEvent({
      type: "run_complete",
      ts: 7,
      final_output: value,
      turn_count: 1,
      response_count: 0,
      shim_state: {},
    });

    await expect(run).resolves.toEqual({
      status: "completed",
      finalOutput: value,
      turnCount: 1,
      responseCount: 0,
    });
    expect(tools.at(-1)).toMatchObject({
      phase: "end",
      result: value,
      call: { arguments: { value } },
    });
    expect(agents.at(-1)).toMatchObject({ phase: "end", finalOutput: value });
    expect(text).toEqual([]);
    child.emitClose();
  });

  test.each([
    2500, 50_000,
  ])("preserves complete text with chunk size %i", async (chunkSize: number): Promise<void> => {
    const child = new FakeChild();
    const chunks: TextChunk[] = [];
    const value = `${"界🧪".repeat(4000)}\n tail\t `;
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onAssistantText: (chunk: TextChunk): void => {
        chunks.push(chunk);
      },
    });
    child.emitReady();
    const session = await sessionPromise;
    const run = session.run("go");
    child.emitEvent({ type: "run_start", ts: 2, prompt: "go" });
    for (let offset = 0; offset < value.length; offset += chunkSize) {
      child.emitModel("text_delta", {
        text: value.slice(offset, offset + chunkSize),
      });
    }
    child.emitEvent({
      type: "run_complete",
      ts: 4,
      final_output: value,
      turn_count: 1,
      response_count: 0,
      shim_state: {},
    });

    await expect(run).resolves.toMatchObject({ finalOutput: value });
    expect(chunks.at(-1)).toMatchObject({ text: value, done: true });
    expect(chunks.map((chunk) => chunk.delta).join("")).toBe(value);
    child.emitClose();
  });

  test("closes streamed text without rendering a structured final result", async () => {
    const child = new FakeChild();
    const chunks: TextChunk[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      onAssistantText: (chunk: TextChunk): void => {
        chunks.push(chunk);
      },
    });
    child.emitReady();
    const session = await sessionPromise;
    const run = session.run("go");
    child.emitEvent({ type: "run_start", ts: 2, prompt: "go" });
    child.emitModel("text_delta", { text: "existing text" });
    child.emitEvent({
      type: "run_complete",
      ts: 4,
      final_output: { answer: 42 },
      turn_count: 1,
      response_count: 0,
      shim_state: {},
    });

    await expect(run).resolves.toMatchObject({ finalOutput: { answer: 42 } });
    expect(chunks.at(-1)).toMatchObject({ text: "existing text", done: true });
    child.emitClose();
  });
});

test("raw events retain empty deltas, structured reasoning and future kinds without rendering", async () => {
  const child = new FakeChild();
  const raw: unknown[] = [];
  const chunks: TextChunk[] = [];
  const sessionPromise = attachAgentSession(child, {
    backend: { command: "fake" },
    onEvent: (event: BridgeEvent): void => {
      raw.push(event);
    },
    onAssistantText: (chunk: TextChunk): void => {
      chunks.push(chunk);
    },
    onReasoningText: (chunk: TextChunk): void => {
      chunks.push(chunk);
    },
  });
  child.emitReady();
  const session = await sessionPromise;
  const run = session.run("go");
  child.emitEvent({ type: "run_start", ts: 1, prompt: "go" });
  child.emitModel("text_delta", { text: "" });
  child.emitModel("reasoning", {
    reasoning: { summary: ["界\n"], signature: "sig" },
  });
  child.emitModel("future_model", {
    text: "must not render",
    provider_extra: { fields: [null, true] },
  });
  child.emitNative("future_native", { content: "must not render" });
  child.emitEvent({
    type: "run_complete",
    ts: 3,
    final_output: null,
    turn_count: 1,
    response_count: 0,
    shim_state: {},
  });
  await run;
  expect(chunks).toEqual([]);
  expect(raw.slice(2, -1)).toMatchObject([
    { event: { payload: { event: { kind: "text_delta", text: "" } } } },
    {
      event: {
        payload: {
          event: { reasoning: { summary: ["界\n"], signature: "sig" } },
        },
      },
    },
    {
      event: {
        payload: {
          event: {
            kind: "future_model",
            provider_extra: { fields: [null, true] },
          },
        },
      },
    },
    {
      event: { type: "future_native", payload: { content: "must not render" } },
    },
  ]);
  child.emitClose();
});

test("model errors and completions do not settle a run or unrelated configure command", async () => {
  const child = new FakeChild();
  const sessionPromise = attachAgentSession(child, {
    backend: { command: "fake" },
  });
  child.emitReady();
  const session = await sessionPromise;
  let settled = false;
  const run = session.run("go").then((value) => {
    settled = true;
    return value;
  });
  child.emitEvent({ type: "run_start", ts: 1, prompt: "go" });
  const configure = session.configure({ model: "next" });
  child.emitModel("error", {
    error: { type: "ValueError", message: "provider error" },
  });
  child.emitModel("completed");
  await Promise.resolve();
  expect(settled).toBe(false);
  child.emitEvent({
    type: "error",
    ts: 2,
    scope: "command",
    message: "configure failed",
  });
  await expect(configure).rejects.toThrow("configure failed");
  expect(settled).toBe(false);
  child.emitModel("text_delta", { text: "continued" });
  child.emitEvent({
    type: "run_complete",
    ts: 3,
    final_output: "continued",
    turn_count: 1,
    response_count: 1,
    shim_state: {},
  });
  await expect(run).resolves.toMatchObject({
    status: "completed",
    finalOutput: "continued",
  });
  child.emitClose();
});

test("source tool ids and arguments remain intact while display flags are derived", async () => {
  const child = new FakeChild();
  const tools: ToolActivity[] = [];
  const sessionPromise = attachAgentSession(child, {
    backend: { command: "fake" },
    onToolActivity: (event: ToolActivity): void => {
      tools.push(event);
    },
  });
  child.emitReady();
  await sessionPromise;
  const source = {
    task_name: "Child",
    task_id: "child",
    parent_task_id: "root",
    is_root: false,
    tool_call: {
      id: "",
      type: "function",
      function: { name: "write_plan", arguments: "invalid-json" },
    },
    is_delegation: true,
  };
  child.emitNative("tool_start", source);
  child.emitNative("tool_end", {
    ...source,
    result: { complete: true },
    ok: true,
    error: null,
  });
  expect(tools).toMatchObject([
    {
      phase: "start",
      call: {
        callId: "",
        taskId: "child",
        arguments: "invalid-json",
        isPlan: true,
        isDelegation: true,
      },
    },
    {
      phase: "end",
      call: { callId: "", arguments: "invalid-json" },
      result: { complete: true },
    },
  ]);
  child.emitClose();
});
