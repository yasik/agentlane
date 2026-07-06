import type { BridgeProcessOptions } from "./process.ts";

/**
 * Backend launched through the packaged Python bridge entrypoint.
 *
 * This is the normal app-facing form. It launches:
 * `uv run --project <projectDir> python -m agentlane_process_bridge --app <app>`.
 */
export type PythonBackendSpec = {
  /** Factory reference as `module:attribute`, for example `my_app.backend:create_backend`. */
  app: string;

  /** Directory of the uv project that owns the backend. Defaults to `process.cwd()`. */
  projectDir?: string;

  /** Environment for the backend process. Defaults to `process.env`. */
  env?: NodeJS.ProcessEnv;
};

/**
 * Backend declaration accepted by `createAgentSession`.
 *
 * The `app` arm is the packaged Python entrypoint. The `command` arm reuses the
 * low-level process options for tests, custom launchers, and existing bridge
 * processes that already speak the stdio protocol.
 */
export type BackendSpec = PythonBackendSpec | BridgeProcessOptions;

/** Resolve an app-facing backend declaration into child-process options. */
export function resolveBackendSpec(backend: BackendSpec): BridgeProcessOptions {
  const record = backend as Record<string, unknown>;
  const hasApp = typeof record.app === "string";
  const hasCommand = typeof record.command === "string";

  // TypeScript's union checks do not catch every constructed object. Runtime
  // validation keeps ambiguous launch specs from silently choosing one arm.
  if (hasApp && hasCommand) {
    throw new Error("Backend spec must not include both app and command.");
  }

  if (!hasApp && !hasCommand) {
    throw new Error("Backend spec must include app or command.");
  }

  if (hasCommand) {
    return backend as BridgeProcessOptions;
  }

  const spec = backend as PythonBackendSpec;
  const projectDir = spec.projectDir ?? process.cwd();
  return {
    command: "uv",
    args: [
      "run",
      "--project",
      projectDir,
      "python",
      "-m",
      "agentlane_process_bridge",
      "--app",
      spec.app,
    ],
    cwd: projectDir,
    env: spec.env ?? process.env,
  };
}
