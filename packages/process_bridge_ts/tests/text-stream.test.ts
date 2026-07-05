import { describe, expect, test } from "bun:test";
import type { TextChunk } from "../src/session-types.ts";
import { TextStreamTracker } from "../src/text-stream.ts";

describe("text stream tracker", () => {
  test("emits one done chunk for an opened assistant segment", () => {
    const chunks: TextChunk[] = [];
    const tracker = new TextStreamTracker(
      {
        onAssistantText: (chunk: TextChunk): void => {
          chunks.push(chunk);
        },
      },
      "immediate",
    );

    tracker.push("assistant", "hello");
    tracker.complete("hello");

    expect(chunks).toEqual([
      { delta: "hello", text: "hello", segment: 1, done: false },
      { delta: "", text: "hello", segment: 1, done: true },
    ]);
  });

  test("interrupts reasoning before opening assistant text", () => {
    const reasoning: TextChunk[] = [];
    const assistant: TextChunk[] = [];
    const tracker = new TextStreamTracker(
      {
        onAssistantText: (chunk: TextChunk): void => {
          assistant.push(chunk);
        },
        onReasoningText: (chunk: TextChunk): void => {
          reasoning.push(chunk);
        },
      },
      "immediate",
    );

    tracker.push("reasoning", "think");
    tracker.push("assistant", "answer");
    tracker.complete("answer");

    expect(reasoning.at(-1)).toEqual({
      delta: "",
      text: "think",
      segment: 1,
      done: true,
    });
    expect(assistant.at(-1)).toEqual({
      delta: "",
      text: "answer",
      segment: 2,
      done: true,
    });
  });

  test("synthesizes final assistant text when no segment is open", () => {
    const chunks: TextChunk[] = [];
    const tracker = new TextStreamTracker(
      {
        onAssistantText: (chunk: TextChunk): void => {
          chunks.push(chunk);
        },
      },
      "immediate",
    );

    tracker.complete("final");

    expect(chunks).toEqual([
      { delta: "final", text: "final", segment: 1, done: true },
    ]);
  });
});
