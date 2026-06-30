import { describe, expect, test } from "bun:test";
import { decodeBridgeEventLine, KNOWN_EVENT_TYPES } from "../src/decoders.ts";
import { encodeBridgeCommand, PROTOCOL_VERSION } from "../src/protocol.ts";

describe("protocol commands", () => {
  test("encodes exactly one versioned NDJSON command", () => {
    const line = encodeBridgeCommand({ type: "prompt", text: "hello" });
    expect(line.endsWith("\n")).toBe(true);
    expect(line.split("\n")).toHaveLength(2);
    expect(JSON.parse(line)).toEqual({
      protocol_version: PROTOCOL_VERSION,
      type: "prompt",
      text: "hello",
    });
  });
});

describe("event decoding", () => {
  test("decodes known event payloads", () => {
    const decoded = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "error",
        ts: 1,
        message: "bad",
        scope: "command",
      }),
    );

    expect(decoded?.fallbacks).toEqual([]);
    expect(decoded?.event).toMatchObject({
      protocol_version: "1.0",
      type: "error",
      ts: 1,
      message: "bad",
      scope: "command",
    });
  });

  test("decodes llm_end token usage and tolerates null usage", () => {
    const base = {
      protocol_version: "1.0",
      type: "llm_end",
      ts: 1,
      task_id: "task",
      parent_task_id: null,
      is_root: true,
      is_subagent: false,
      agent: "Root",
      output_preview: "done",
    };

    const withUsage = decodeBridgeEventLine(
      JSON.stringify({
        ...base,
        usage: {
          prompt_tokens: 1200,
          completion_tokens: 340,
          total_tokens: 1540,
        },
      }),
    );
    const withoutUsage = decodeBridgeEventLine(
      JSON.stringify({ ...base, usage: null }),
    );

    expect(withUsage?.fallbacks).toEqual([]);
    expect(withUsage?.event).toMatchObject({
      type: "llm_end",
      usage: {
        prompt_tokens: 1200,
        completion_tokens: 340,
        total_tokens: 1540,
      },
    });
    expect(withoutUsage?.fallbacks).toEqual([]);
    expect(withoutUsage?.event).toMatchObject({ type: "llm_end", usage: null });
  });

  test("records missing typed fields as fallbacks", () => {
    const decoded = decodeBridgeEventLine(
      JSON.stringify({ type: "run_start", ts: 1 }),
    );

    expect(decoded?.event.type).toBe("run_start");
    expect(decoded?.fallbacks).toEqual(["protocol_version", "prompt"]);
  });

  test("records missing raw fields as fallbacks", () => {
    const toolStart = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "tool_start",
        ts: 1,
        task_id: "task",
        parent_task_id: null,
        is_root: true,
        is_subagent: false,
        agent: "agent",
        tool: "write",
        tool_call_id: "call_1",
        is_plan: false,
        is_delegation: false,
      }),
    );
    const planUpdate = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "plan_updated",
        ts: 1,
        task_id: "task",
        parent_task_id: null,
        is_root: true,
        is_subagent: false,
        agent: "agent",
        tool_call_id: "call_1",
        explanation: null,
        title: null,
      }),
    );

    expect(toolStart?.fallbacks).toContain("arguments");
    expect(planUpdate?.fallbacks).toEqual(["steps", "raw"]);
  });

  test("records approval request payload drift as nested fallbacks", () => {
    const decoded = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "approval_request",
        ts: 1,
        id: "approval-1",
        request: {},
        reason: null,
      }),
    );

    expect(decoded?.event.type).toBe("approval_request");
    expect(decoded?.fallbacks).toContain("request.tool_name");
    expect(decoded?.fallbacks).toContain("request.metadata");
  });

  test("decodes unknown events without crashing", () => {
    const decoded = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "new_event",
        ts: 1,
        value: true,
      }),
    );

    expect(decoded?.event).toMatchObject({
      type: "unknown_event",
      event_type: "new_event",
      payload: {
        protocol_version: "1.0",
        type: "new_event",
        ts: 1,
        value: true,
      },
    });
  });

  test("rejects unsupported protocol major versions", () => {
    expect(
      decodeBridgeEventLine(
        JSON.stringify({ protocol_version: "2.0", type: "ready", ts: 1 }),
      ),
    ).toBeNull();
  });

  test("known event list includes lifecycle and approval events", () => {
    expect(KNOWN_EVENT_TYPES).toContain("approval_request");
    expect(KNOWN_EVENT_TYPES).toContain("run_complete");
  });
});
