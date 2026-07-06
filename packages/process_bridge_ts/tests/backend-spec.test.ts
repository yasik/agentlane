import { describe, expect, test } from "bun:test";
import { resolveBackendSpec } from "../src/backend-spec.ts";

describe("backend specs", () => {
  test("resolves app references through the packaged Python entrypoint", () => {
    const options = resolveBackendSpec({
      app: "demo.backend:create_backend",
      projectDir: "/workspace/app",
      env: { DEMO: "1" },
    });

    expect(options).toEqual({
      command: "uv",
      args: [
        "run",
        "--project",
        "/workspace/app",
        "python",
        "-m",
        "agentlane_process_bridge",
        "--app",
        "demo.backend:create_backend",
      ],
      cwd: "/workspace/app",
      env: { DEMO: "1" },
    });
  });

  test("passes raw process options through unchanged", () => {
    const backend = {
      command: "python",
      args: ["backend.py"],
      cwd: "/tmp/app",
    };

    expect(resolveBackendSpec(backend)).toBe(backend);
  });

  test("rejects ambiguous backend specs", () => {
    expect(() =>
      resolveBackendSpec({
        app: "demo.backend:create_backend",
        command: "python",
      }),
    ).toThrow("both app and command");
    expect(() => resolveBackendSpec({} as never)).toThrow("app or command");
  });
});
