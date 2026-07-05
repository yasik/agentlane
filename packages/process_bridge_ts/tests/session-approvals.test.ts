import { describe, expect, test } from "bun:test";
import type {
  ApprovalRequest,
  ApprovalResolution,
} from "../src/session-types.ts";
import { FakeChild } from "./session-test-helpers.ts";
import { attachAgentSession } from "./session-test-support.ts";

describe("agent session approvals", () => {
  test("sends policy decisions and marks app-decided resolutions", async () => {
    const child = new FakeChild();
    const resolutions: ApprovalResolution[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      approvals: (request: ApprovalRequest): boolean => {
        expect(request.request.tool_name).toBe("write");
        return true;
      },
      onApprovalResolved: (resolution: ApprovalResolution): void => {
        resolutions.push(resolution);
      },
    });
    child.emitReady();
    await sessionPromise;

    child.emitApprovalRequest("approval-1");
    await tick();
    expect(child.commands().at(-1)).toMatchObject({
      type: "approve",
      id: "approval-1",
      allowed: true,
    });

    child.emitApprovalResolved("approval-1", true);

    expect(resolutions).toHaveLength(1);
    expect(resolutions[0]?.decidedByApp).toBe(true);
  });

  test("denies visibly when no approval policy is configured", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
    });
    child.emitReady();
    await sessionPromise;

    child.emitApprovalRequest("approval-1");
    await tick();

    expect(child.commands().at(-1)).toMatchObject({
      type: "approve",
      id: "approval-1",
      allowed: false,
      reason: "No approval policy configured.",
    });
  });

  test("aborts pending approval policies when backend resolves first", async () => {
    const child = new FakeChild();
    const captured: AbortSignal[] = [];
    const sessionPromise = attachAgentSession(child, {
      backend: { command: "fake" },
      approvals: (request: ApprovalRequest): Promise<boolean> => {
        captured.push(request.signal);
        return new Promise((resolve) => setTimeout(() => resolve(true), 10));
      },
    });
    child.emitReady();
    await sessionPromise;

    child.emitApprovalRequest("approval-1");
    await tick();
    child.emitApprovalResolved("approval-1", false);
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(captured[0]?.aborted).toBe(true);
    expect(
      child.commands().filter((command) => command.type === "approve"),
    ).toEqual([]);
  });
});

function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
