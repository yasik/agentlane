import { describe, expect, test } from "bun:test";
import {
  type AgentSession,
  ConfigureError,
  type SessionClose,
  SessionClosedError,
} from "../src/session-types.ts";
import { FakeChild } from "./session-test-helpers.ts";
import { attachAgentSession } from "./session-test-support.ts";

type ModelConfig = {
  model: string;
  attributes?: Record<string, unknown>;
};

type ModelPatch = {
  model?: string | null;
};

function assertConfigPatchTypes(
  session: AgentSession<ModelConfig, ModelPatch>,
): void {
  void session.configure({ model: "openai/gpt-5.5" });
  void session.configure({ model: null });
  // @ts-expect-error Unknown config patch keys must fail on closed app shapes.
  void session.configure({ typo: "openai/gpt-5.5" });
}

void assertConfigPatchTypes;

describe("agent session config", () => {
  test("captures ready config without firing the change callback", async () => {
    const child = new FakeChild();
    const changes: ModelConfig[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onConfigChanged: (config: Readonly<ModelConfig>): void => {
        changes.push({ ...config });
      },
    });

    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    expect(session.config).toEqual({ model: "openai/gpt-5.5" });
    expect(changes).toEqual([]);
  });

  test("configure resolves with the applied document and updates the cache", async () => {
    const child = new FakeChild();
    const changes: ModelConfig[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onConfigChanged: (config: Readonly<ModelConfig>): void => {
        changes.push({ ...config });
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "anthropic/claude-opus-4-8" });

    expect(child.commands().at(-1)).toMatchObject({
      type: "configure",
      patch: { model: "anthropic/claude-opus-4-8" },
    });
    child.emitEvent({
      type: "config",
      ts: 2,
      ok: true,
      config: { model: "anthropic/claude-opus-4-8" },
      error: null,
    });

    await expect(configure).resolves.toEqual({
      model: "anthropic/claude-opus-4-8",
    });
    expect(session.config).toEqual({ model: "anthropic/claude-opus-4-8" });
    expect(changes).toEqual([{ model: "anthropic/claude-opus-4-8" }]);
  });

  test("configure rejects all backend failure codes", async () => {
    for (const code of [
      "invalid",
      "unsupported",
      "rejected",
      "internal",
    ] as const) {
      const child = new FakeChild();
      const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(
        child,
        {
          backend: { command: "fake" },
          decodeConfig: decodeModelConfig,
        },
      );
      emitReady(child, { model: "openai/gpt-5.5" });
      const session = await sessionPromise;

      const configure = session.configure({ model: "openai/gpt-9" });
      child.emitEvent({
        type: "config",
        ts: 2,
        ok: false,
        config: { model: "openai/gpt-5.5" },
        error: { code, message: `${code} failure` },
      });

      await expect(configure).rejects.toBeInstanceOf(ConfigureError);
      await configure.catch((error: unknown): void => {
        expect(error).toMatchObject({ code, message: `${code} failure` });
      });
      expect(session.config).toEqual({ model: "openai/gpt-5.5" });
    }
  });

  test("failed configure re-syncs the cache before rejecting", async () => {
    const child = new FakeChild();
    const changes: ModelConfig[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onConfigChanged: (config: Readonly<ModelConfig>): void => {
        changes.push({ ...config });
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "openai/gpt-9" });
    child.emitEvent({
      type: "config",
      ts: 2,
      ok: false,
      config: { model: "openai/gpt-5.5", attributes: { effort: "medium" } },
      error: { code: "rejected", message: "Unknown model" },
    });

    await expect(configure).rejects.toBeInstanceOf(ConfigureError);
    expect(session.config).toEqual({
      model: "openai/gpt-5.5",
      attributes: { effort: "medium" },
    });
    expect(changes).toEqual([
      { model: "openai/gpt-5.5", attributes: { effort: "medium" } },
    ]);
  });

  test("failed configure without a snapshot keeps the last known config", async () => {
    const child = new FakeChild();
    const changes: ModelConfig[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onConfigChanged: (config: Readonly<ModelConfig>): void => {
        changes.push({ ...config });
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "openai/gpt-9" });
    child.emitEvent({
      type: "config",
      ts: 2,
      ok: false,
      config: null,
      error: { code: "internal", message: "backend failed" },
    });

    await expect(configure).rejects.toBeInstanceOf(ConfigureError);
    expect(session.config).toEqual({ model: "openai/gpt-5.5" });
    expect(changes).toEqual([]);
  });

  test("command-scoped errors reject pending configure", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "openai/gpt-9" });
    child.emitEvent({
      type: "error",
      ts: 2,
      scope: "command",
      message: "Unknown command: configure",
    });

    await expect(configure).rejects.toMatchObject({
      code: "unsupported",
      message: "Unknown command: configure",
    });
  });

  test("undefined patch values reject before sending a command", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    await expect(
      session.configure({ model: undefined } as unknown as ModelPatch),
    ).rejects.toMatchObject({ code: "invalid" });
    expect(child.commands()).toEqual([]);
  });

  test("reset re-announces config before resolving reset", async () => {
    const child = new FakeChild();
    const changes: ModelConfig[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onConfigChanged: (config: Readonly<ModelConfig>): void => {
        changes.push({ ...config });
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const reset = session.reset();
    child.emitEvent({
      type: "reset",
      ts: 2,
      config: { model: "anthropic/claude-opus-4-8" },
    });

    await expect(reset).resolves.toBeUndefined();
    expect(session.config).toEqual({ model: "anthropic/claude-opus-4-8" });
    expect(changes).toEqual([{ model: "anthropic/claude-opus-4-8" }]);
  });

  test("rapid configure calls settle in FIFO order", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const first = session.configure({ model: "openai/gpt-5.5-mini" });
    const second = session.configure({ model: "anthropic/claude-opus-4-8" });

    expect(child.commands().map((command) => command.type)).toEqual([
      "configure",
      "configure",
    ]);
    child.emitEvent({
      type: "config",
      ts: 2,
      ok: true,
      config: { model: "openai/gpt-5.5-mini" },
      error: null,
    });
    child.emitEvent({
      type: "config",
      ts: 3,
      ok: true,
      config: { model: "anthropic/claude-opus-4-8" },
      error: null,
    });

    await expect(first).resolves.toEqual({ model: "openai/gpt-5.5-mini" });
    await expect(second).resolves.toEqual({
      model: "anthropic/claude-opus-4-8",
    });
    expect(session.config).toEqual({ model: "anthropic/claude-opus-4-8" });
  });

  test("decodeConfig failures close the session and reject configure", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onDiagnostic: (): void => undefined,
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "bad" });
    child.emitEvent({
      type: "config",
      ts: 2,
      ok: true,
      config: { model: 123 },
      error: null,
    });

    await expect(configure).rejects.toBeInstanceOf(SessionClosedError);
    expect(closes[0]?.reason).toBe("protocol-error");
    expect(child.killed).toContain("SIGKILL");
  });

  test("ready config decode failures reject startup without onClose", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onDiagnostic: (): void => undefined,
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });

    emitReady(child, { model: 123 } as unknown as ModelConfig);

    await expect(sessionPromise).rejects.toThrow("model must be a string");
    expect(closes).toEqual([]);
    expect(child.killed).toContain("SIGKILL");
  });

  test("unexpected config events close the session", async () => {
    const child = new FakeChild();
    const closes: SessionClose[] = [];
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
      onDiagnostic: (): void => undefined,
      onClose: (close: SessionClose): void => {
        closes.push(close);
      },
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    await sessionPromise;

    child.emitEvent({
      type: "config",
      ts: 2,
      ok: true,
      config: { model: "anthropic/claude-opus-4-8" },
      error: null,
    });

    expect(closes[0]?.reason).toBe("protocol-error");
    expect(child.killed).toContain("SIGKILL");
  });

  test("in-flight configure rejects when the session closes", async () => {
    const child = new FakeChild();
    const sessionPromise = attachAgentSession<ModelConfig, ModelPatch>(child, {
      backend: { command: "fake" },
      decodeConfig: decodeModelConfig,
    });
    emitReady(child, { model: "openai/gpt-5.5" });
    const session = await sessionPromise;

    const configure = session.configure({ model: "anthropic/claude-opus-4-8" });
    child.emitClose(1, null);

    await expect(configure).rejects.toBeInstanceOf(SessionClosedError);
  });
});

function emitReady(child: FakeChild, config: ModelConfig): void {
  child.emitEvent({
    type: "ready",
    ts: 1,
    version: "0.1.0",
    package: "agentlane-process-bridge",
    config,
  });
}

function decodeModelConfig(raw: Record<string, unknown>): ModelConfig {
  if (typeof raw.model !== "string") {
    throw new Error("model must be a string");
  }

  return { ...raw, model: raw.model };
}
