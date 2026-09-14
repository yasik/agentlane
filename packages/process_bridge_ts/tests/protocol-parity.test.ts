import { describe, expect, test } from "bun:test";
import { decodeBridgeEventLine, KNOWN_EVENT_TYPES } from "../src/decoders.ts";
import { KNOWN_COMMAND_TYPES } from "../src/protocol.ts";
import { isNativeEvent, NATIVE_EVENT_SCHEMAS } from "../src/protocol-native.ts";

const fixtureUrl: URL = new URL(
  "../../process_bridge/fixtures/protocol/events.json",
  import.meta.url,
);

describe("protocol fixtures", () => {
  test("command type list covers known command union", () => {
    expectSameSet(
      ["prompt", "approve", "cancel", "configure", "reset", "shutdown"],
      [...KNOWN_COMMAND_TYPES],
      "bridge command union",
      "TypeScript command type list",
    );
  });

  test("decode every fixture strictly", async () => {
    const fixtures = (await Bun.file(fixtureUrl).json()) as unknown[];
    const fixtureTypes = fixtures.map((fixture: unknown): string => {
      const decoded = decodeBridgeEventLine(JSON.stringify(fixture));
      expect(fixture).toEqual(decoded);
      return decoded.type;
    });

    expectSameSet(
      [...KNOWN_EVENT_TYPES],
      fixtureTypes,
      "TypeScript decoder event types",
      "protocol fixtures",
    );
  });

  test("fixtures cover every native kind, model kind, approval status and preserve usage extras", async () => {
    const fixtures = (await Bun.file(fixtureUrl).json()) as unknown[];
    const events = fixtures
      .map((fixture) => decodeBridgeEventLine(JSON.stringify(fixture)))
      .filter((event) => event.type === "run_event")
      .map((event) => event.event);
    expectSameSet(
      Object.keys(NATIVE_EVENT_SCHEMAS),
      events.map((event) => event.type),
      "native schemas",
      "native fixtures",
    );
    expectSameSet(
      [
        "text_delta",
        "reasoning",
        "tool_call_arguments_delta",
        "provider",
        "completed",
        "error",
      ],
      events
        .filter((event) => isNativeEvent(event, "model_stream"))
        .map((event) => event.payload.event.kind),
      "model kinds",
      "model fixtures",
    );
    expectSameSet(
      ["pending", "resolved"],
      events
        .filter((event) => isNativeEvent(event, "tool_approval"))
        .map((event) => event.payload.event.record.status),
      "approval statuses",
      "approval fixtures",
    );
    const llm = events.find((event) => isNativeEvent(event, "llm_end"));
    expect(llm?.payload.response.usage).toBeNull();
  });

  test("missing required fixture fields fail loudly with named paths", () => {
    expect(() =>
      decodeBridgeEventLine(
        JSON.stringify({
          protocol_version: "1.0",
          type: "run_event",
          ts: 1,
          event: { type: "tool_end", payload: { kind: "tool_end" } },
        }),
      ),
    ).toThrow("tool_call");
  });
});

function expectSameSet(
  expected: string[],
  actual: string[],
  expectedName: string,
  actualName: string,
): void {
  const expectedSet = new Set(expected);
  const actualSet = new Set(actual);
  const missing = [...expectedSet].filter((type) => !actualSet.has(type));
  const extra = [...actualSet].filter((type) => !expectedSet.has(type));

  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `${actualName} do not match ${expectedName}. ` +
        `Missing: ${missing.length === 0 ? "none" : missing.join(", ")}. ` +
        `Extra: ${extra.length === 0 ? "none" : extra.join(", ")}.`,
    );
  }
}

test("retains populated source usage and SDK extensions", async () => {
  const fixtures = (await Bun.file(fixtureUrl).json()) as unknown[];
  const source = fixtures
    .map((fixture) => decodeBridgeEventLine(JSON.stringify(fixture)))
    .find(
      (event) =>
        event.type === "run_event" && isNativeEvent(event.event, "llm_end"),
    );
  if (source?.type !== "run_event" || !isNativeEvent(source.event, "llm_end"))
    throw new Error("Missing LLM response fixture");
  source.event.payload.response.usage = {
    prompt_tokens: 12,
    completion_tokens: 3,
    total_tokens: 15,
    prompt_tokens_details: { cached_tokens: 9, sdk_extra: [null, "界"] },
  };
  expect(decodeBridgeEventLine(JSON.stringify(source))).toEqual(source);
});

test("keeps own magic keys in known provider records and approval metadata", async () => {
  const fixtures = (await Bun.file(fixtureUrl).json()) as unknown[];
  const events = fixtures.map((fixture) =>
    decodeBridgeEventLine(JSON.stringify(fixture)),
  );
  const extras = JSON.parse(
    '{"__proto__":{"polluted":true},"constructor":{"source":true},"nested":{"__proto__":{"kept":true}}}',
  ) as Record<string, unknown>;
  for (const event of events) {
    if (event.type !== "run_event") continue;
    if (isNativeEvent(event.event, "model_stream"))
      event.event.payload.event.raw = extras;
    if (isNativeEvent(event.event, "tool_approval"))
      event.event.payload.event.record.request.metadata = extras;
    const line = JSON.stringify(event);
    expect(JSON.stringify(decodeBridgeEventLine(line))).toBe(line);
  }
  expect(({} as Record<string, unknown>).polluted).toBeUndefined();
});
