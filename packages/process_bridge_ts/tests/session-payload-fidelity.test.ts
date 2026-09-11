import { describe, expect, test } from "bun:test";
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
      is_subagent: false,
      agent: "Root",
    };
    child.emitEvent({ type: "agent_start", ts: 3, ...lineage, next_turn: 1 });
    child.emitEvent({
      type: "tool_start",
      ts: 4,
      ...lineage,
      tool: "read",
      tool_call_id: "call",
      arguments: { value },
      is_plan: false,
      is_delegation: false,
    });
    child.emitEvent({
      type: "tool_end",
      ts: 5,
      ...lineage,
      tool: "read",
      tool_call_id: "call",
      result: value,
      ok: true,
      error: null,
      is_plan: false,
      is_delegation: false,
    });
    child.emitEvent({
      type: "agent_end",
      ts: 6,
      ...lineage,
      final_output: value,
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
      child.emitEvent({
        type: "assistant_delta",
        ts: 3,
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
    child.emitEvent({ type: "assistant_delta", ts: 3, text: "existing text" });
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
