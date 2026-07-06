import { describe, expect, test } from "bun:test";
import { decodeBridgeEventLine, KNOWN_EVENT_TYPES } from "../src/decoders.ts";
import { KNOWN_COMMAND_TYPES } from "../src/protocol.ts";

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
      return decoded.type;
    });

    expectSameSet(
      [...KNOWN_EVENT_TYPES],
      fixtureTypes,
      "TypeScript decoder event types",
      "protocol fixtures",
    );
  });

  test("missing required fixture fields fail loudly with named paths", () => {
    expect(() =>
      decodeBridgeEventLine(
        JSON.stringify({
          protocol_version: "1.0",
          type: "tool_end",
          ts: 1,
        }),
      ),
    ).toThrow("tool_call_id");
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
