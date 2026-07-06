import { describe, expect, test } from "bun:test";
import {
  BridgeDecodeError,
  decodeBridgeEventLine,
  KNOWN_EVENT_TYPES,
  tryDecodeBridgeEventLine,
} from "../src/decoders.ts";
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

  test("encodes configure patches as opaque JSON objects", () => {
    const line = encodeBridgeCommand({
      type: "configure",
      patch: { model: "anthropic/claude-opus-4-8" },
    });

    expect(JSON.parse(line)).toEqual({
      protocol_version: PROTOCOL_VERSION,
      type: "configure",
      patch: { model: "anthropic/claude-opus-4-8" },
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

    expect(decoded).toMatchObject({
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

    expect(withUsage).toMatchObject({
      type: "llm_end",
      usage: {
        prompt_tokens: 1200,
        completion_tokens: 340,
        total_tokens: 1540,
      },
    });
    expect(withoutUsage).toMatchObject({ type: "llm_end", usage: null });
  });

  test("rejects missing typed fields", () => {
    const error = decodeErrorFor(
      JSON.stringify({ protocol_version: "1.0", type: "run_start", ts: 1 }),
    );

    expect(error.fields).toEqual(["prompt"]);
  });

  test("rejects missing raw fields", () => {
    const toolStartError = decodeErrorFor(
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
    const planUpdateError = decodeErrorFor(
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

    expect(toolStartError.fields).toContain("arguments");
    expect(planUpdateError.fields).toEqual(["raw", "steps"]);
  });

  test("rejects approval request payload drift with nested field paths", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "approval_request",
        ts: 1,
        id: "approval-1",
        request: {},
        reason: null,
      }),
    );

    expect(error.fields).toContain("request.tool_name");
    expect(error.fields).toContain("request.metadata");
  });

  test("rejects extra fields on known event payloads", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "run_start",
        ts: 1,
        prompt: "go",
        unexpected: true,
      }),
    );

    expect(error.fields).toEqual(["event"]);
  });

  test("rejects extra fields on nested protocol payloads", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "llm_end",
        ts: 1,
        task_id: "task",
        parent_task_id: null,
        is_root: true,
        is_subagent: false,
        agent: "Root",
        output_preview: "done",
        usage: {
          prompt_tokens: 1200,
          completion_tokens: 340,
          total_tokens: 1540,
          cached_tokens: 100,
        },
      }),
    );

    expect(error.fields).toEqual(["usage"]);
  });

  test("decodes ready, reset, and config documents", () => {
    const ready = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "ready",
        ts: 1,
        version: "0.1.0",
        package: "agentlane-process-bridge",
        config: { model: "openai/gpt-5.5" },
      }),
    );
    const config = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 2,
        ok: true,
        config: { model: "anthropic/claude-opus-4-8" },
        error: null,
      }),
    );
    const reset = decodeBridgeEventLine(
      JSON.stringify({
        protocol_version: "1.0",
        type: "reset",
        ts: 3,
        config: { model: "anthropic/claude-opus-4-8" },
      }),
    );

    expect(ready).toMatchObject({
      type: "ready",
      config: { model: "openai/gpt-5.5" },
    });
    expect(config).toMatchObject({ type: "config", ok: true });
    expect(reset).toMatchObject({
      type: "reset",
      config: { model: "anthropic/claude-opus-4-8" },
    });
  });

  test("decodes every configure failure code", () => {
    for (const code of ["invalid", "unsupported", "rejected", "internal"]) {
      const decoded = decodeBridgeEventLine(
        JSON.stringify({
          protocol_version: "1.0",
          type: "config",
          ts: 1,
          ok: false,
          config: { model: "openai/gpt-5.5" },
          error: { code, message: `${code} failure` },
        }),
      );

      expect(decoded).toMatchObject({
        type: "config",
        ok: false,
        error: { code, message: `${code} failure` },
      });
    }
  });

  test("rejects invalid config settlement invariants", () => {
    const successWithoutConfig = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 1,
        ok: true,
        config: null,
        error: null,
      }),
    );
    const failureWithoutError = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "config",
        ts: 1,
        ok: false,
        config: null,
        error: null,
      }),
    );

    expect(successWithoutConfig.fields).toContain("config");
    expect(failureWithoutError.fields).toContain("error");
  });

  test("rejects unknown events", () => {
    const error = decodeErrorFor(
      JSON.stringify({
        protocol_version: "1.0",
        type: "new_event",
        ts: 1,
        value: true,
      }),
    );

    expect(error.fields).toEqual(["type"]);
    expect(error.message).toContain("Unknown bridge event type");
  });

  test("rejects unsupported protocol major versions", () => {
    expect(
      tryDecodeBridgeEventLine(
        JSON.stringify({ protocol_version: "2.0", type: "ready", ts: 1 }),
      ),
    ).toBeNull();
  });

  test("known event list includes lifecycle and approval events", () => {
    expect(KNOWN_EVENT_TYPES).toContain("approval_request");
    expect(KNOWN_EVENT_TYPES).toContain("run_complete");
  });
});

function decodeErrorFor(line: string): BridgeDecodeError {
  try {
    decodeBridgeEventLine(line);
  } catch (error) {
    if (error instanceof BridgeDecodeError) return error;

    throw error;
  }

  throw new Error("Expected bridge decode to fail.");
}
