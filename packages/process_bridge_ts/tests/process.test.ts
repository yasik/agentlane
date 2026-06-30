import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { PassThrough } from "node:stream";
import { createBridgeChannel } from "../src/channel.ts";
import { spawnBridgeProcess, wireBridgeProcess } from "../src/process.ts";
import type { BridgeEvent } from "../src/protocol.ts";

describe("process wiring", () => {
  test("routes events, fallbacks, invalid lines, and stderr separately", async () => {
    const stdout = new PassThrough();
    const stderr = new PassThrough();
    const events: BridgeEvent[] = [];
    const fallbacks: string[][] = [];
    const invalid: string[] = [];
    const diagnostics: string[] = [];

    const wiring = wireBridgeProcess(
      { stdout, stderr },
      {
        onEvent: (event: BridgeEvent): void => {
          events.push(event);
        },
        onDecodeFallback: (_type: string, fields: string[]): void => {
          fallbacks.push(fields);
        },
        onInvalidLine: (line: string): void => {
          invalid.push(line);
        },
        onStderr: (line: string): void => {
          diagnostics.push(line);
        },
      },
    );

    stdout.write(
      `${JSON.stringify({
        protocol_version: "1.0",
        type: "run_start",
        ts: 1,
        prompt: "go",
      })}\n`,
    );
    stdout.write(`${JSON.stringify({ type: "run_start", ts: 2 })}\n`);
    stdout.write("not-json\n");
    stderr.write("diagnostic\n");

    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(events.map((event) => event.type)).toEqual(["run_start"]);
    expect(fallbacks).toEqual([["protocol_version", "prompt"]]);
    expect(invalid).toEqual(["not-json"]);
    expect(diagnostics).toEqual(["diagnostic"]);

    wiring.dispose();
    stdout.write(
      `${JSON.stringify({
        protocol_version: "1.0",
        type: "run_start",
        ts: 3,
        prompt: "ignored",
      })}\n`,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(events.map((event) => event.type)).toEqual(["run_start"]);
  });

  test("spawnBridgeProcess drains final stdout before exit callback", async () => {
    const events: BridgeEvent[] = [];

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("child process did not exit")),
        5000,
      );
      spawnBridgeProcess(
        {
          command: process.execPath,
          args: [
            "-e",
            "process.stdout.write(JSON.stringify({protocol_version:'1.0',type:'shutdown',ts:1}) + '\\n');",
          ],
        },
        {
          onEvent: (event: BridgeEvent): void => {
            events.push(event);
          },
          onExit: (): void => {
            clearTimeout(timeout);
            resolve();
          },
          onSpawnError: (error: Error): void => {
            clearTimeout(timeout);
            reject(error);
          },
        },
      );
    });

    expect(events.map((event) => event.type)).toEqual(["shutdown"]);
  });

  test("drives the Python stdio backend through the channel API", async () => {
    const repoRoot = resolve(import.meta.dir, "../../..");
    const events: BridgeEvent[] = [];
    const fallbacks: string[][] = [];
    const invalid: string[] = [];
    const stderr: string[] = [];
    let channel: ReturnType<typeof createBridgeChannel> | null = null;
    let readySeen = false;
    let promptSent = false;

    await new Promise<void>((resolveTest, rejectTest) => {
      let childProcess: ReturnType<typeof spawnBridgeProcess> | null = null;

      const stopChild = (): void => {
        channel?.shutdown();
        childProcess?.kill("SIGKILL");
      };

      const timeout = setTimeout(() => {
        stopChild();
        rejectTest(new Error("Python bridge smoke test timed out"));
      }, 20_000);

      const finish = (assertions: () => void): void => {
        clearTimeout(timeout);
        try {
          assertions();
          resolveTest();
        } catch (error) {
          rejectTest(error);
        }
      };

      const fail = (error: Error): void => {
        clearTimeout(timeout);
        stopChild();
        rejectTest(error);
      };

      const sendPrompt = (): void => {
        if (!readySeen || promptSent || channel === null) return;

        promptSent = true;
        if (!channel.send({ type: "prompt", text: "hello" })) {
          fail(new Error("failed to send prompt to Python bridge"));
        }
      };

      childProcess = spawnBridgeProcess(
        {
          command: "uv",
          args: [
            "run",
            "python",
            "examples/harness/process_bridge_stdio/backend.py",
          ],
          cwd: repoRoot,
        },
        {
          onDecodeFallback: (_type: string, fields: string[]): void => {
            fallbacks.push(fields);
          },
          onEvent: (event: BridgeEvent): void => {
            events.push(event);

            if (event.type === "ready") {
              readySeen = true;
              sendPrompt();
            }

            if (event.type === "run_complete") {
              channel?.shutdown();
            }
          },
          onExit: (
            code: number | null,
            signal: NodeJS.Signals | null,
          ): void => {
            finish(() => {
              expect(code).toBe(0);
              expect(signal).toBeNull();
              expect(fallbacks).toEqual([]);
              expect(invalid).toEqual([]);
              expect(stderr).toEqual([]);
              expect(events.map((event) => event.type)).toEqual([
                "ready",
                "run_start",
                "agent_start",
                "assistant_delta",
                "agent_end",
                "run_complete",
                "shutdown",
              ]);
            });
          },
          onInvalidLine: (line: string): void => {
            invalid.push(line);
          },
          onSpawnError: fail,
          onStderr: (line: string): void => {
            stderr.push(line);
          },
        },
      );

      channel = createBridgeChannel(childProcess, {
        graceMs: 5000,
        onSendError: (message: string): void => {
          fail(new Error(message));
        },
      });
      sendPrompt();
    });
  });
});
