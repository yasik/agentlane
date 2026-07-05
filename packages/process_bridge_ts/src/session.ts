// biome-ignore-all lint/nursery/noExcessiveLinesPerFile: The session controller keeps process lifecycle, command FIFO, and promise settlement together; smaller helpers own config, command-error, and reducer adapter logic.
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { resolveBackendSpec } from "./backend-spec.ts";
import type { BridgeChildLike } from "./channel.ts";
import { createBridgeChannel } from "./channel.ts";
import { BridgeDecodeError } from "./decoders.ts";
import type { BridgeProcessOptions } from "./process.ts";
import { spawnBridgeProcess } from "./process.ts";
import type { BridgeCommand, BridgeEvent } from "./protocol.ts";
import { handlePendingCommandError } from "./session-command-errors.ts";
import { SessionConfigState } from "./session-config.ts";
import {
  createSessionProcessCallbacks,
  createSessionReducerCallbacks,
} from "./session-controller-callbacks.ts";
import { type Deferred, deferred } from "./session-deferred.ts";
import { createSessionHandle } from "./session-handle.ts";
import {
  type PendingCommand,
  PendingCommandQueue,
} from "./session-pending-commands.ts";
import {
  SessionReducer,
  type SessionReducerCallbacks,
} from "./session-reducer.ts";
import {
  type AgentSession,
  type AgentSessionOptions,
  type ReadyInfo,
  type RunResult,
  type SessionClose,
  SessionClosedError,
  type SessionDiagnostic,
  SessionStartError,
  SessionStateError,
} from "./session-types.ts";

/**
 * Spawn the backend, wire stdio, and resolve once `ready` arrives.
 *
 * Startup failures reject with `SessionStartError` before a live session is
 * exposed. After startup, all operation promises settle through the session
 * lifecycle; app code never needs to observe the raw child process.
 */
export function createAgentSession<
  TConfig extends Record<string, unknown> = Record<string, unknown>,
>(options: AgentSessionOptions<TConfig>): Promise<AgentSession<TConfig>> {
  let processOptions: BridgeProcessOptions;
  try {
    processOptions = resolveBackendSpec(options.backend);
  } catch (error) {
    return Promise.reject(
      new SessionStartError(
        error instanceof Error ? error.message : String(error),
      ),
    );
  }

  const controller = new AgentSessionController<TConfig>(options);
  const child = spawnBridgeProcess(
    processOptions,
    controller.processCallbacks(),
  );
  return controller.start(child);
}

/**
 * @internal
 *
 * Owns the live session lifecycle and every promise exposed to app code.
 *
 * The controller keeps transport concerns here: ready gating, command
 * correlation, process exit handling, and operation settlement. Semantic event
 * reduction lives in `SessionReducer`, which keeps UI-facing callbacks separate
 * from child-process bookkeeping.
 */
export class AgentSessionController<
  TConfig extends Record<string, unknown> = Record<string, unknown>,
> {
  private readonly options: AgentSessionOptions<TConfig>;
  private readonly readyDeferred = deferred<AgentSession<TConfig>>();
  private readonly pendingCommands = new PendingCommandQueue();
  private readonly cancelWaiters: Array<Deferred<void>> = [];
  private readonly resetWaiters: Array<Deferred<void>> = [];
  private readonly configState: SessionConfigState<TConfig>;
  private reducer: SessionReducer;
  private channel: ReturnType<typeof createBridgeChannel> | null = null;
  private child:
    | (BridgeChildLike & { kill: (signal?: NodeJS.Signals) => boolean })
    | null = null;
  private readyTimer: ReturnType<typeof setTimeout> | null = null;
  private exitHook: (() => void) | null = null;
  private activeRun: Deferred<RunResult> | null = null;
  private closeDeferred: Deferred<void> | null = null;
  private readyInfo: ReadyInfo | null = null;
  private closed: SessionClose | null = null;
  private startupFailed = false;
  private closeRequested = false;
  private shutdownSeen = false;

  constructor(options: AgentSessionOptions<TConfig>) {
    this.options = options;
    this.configState = new SessionConfigState<TConfig>({
      decodeConfig: options.decodeConfig,
      onConfigChanged: options.onConfigChanged,
      callAppHandler: (handler: string, call: () => void): void => {
        this.callAppHandler(handler, call);
      },
      failProtocol: (message: string, fields: readonly string[]): void => {
        this.failProtocol(message, fields);
      },
    });
    this.reducer = new SessionReducer(this.reducerCallbacks());
  }

  processCallbacks(): {
    onDecodeError: (error: BridgeDecodeError, line: string) => void;
    onEvent: (event: BridgeEvent) => void;
    onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
    onSpawnError: (error: Error) => void;
    onStderr: (line: string) => void;
  } {
    return createSessionProcessCallbacks({
      diagnostic: (diagnostic: SessionDiagnostic): void => {
        this.diagnostic(diagnostic);
      },
      closeForDecodeError: (error: BridgeDecodeError): void => {
        this.closeForDecodeError(error);
      },
      handleEvent: (event: BridgeEvent): void => {
        this.handleEvent(event);
      },
      handleExit: (
        code: number | null,
        signal: NodeJS.Signals | null,
      ): void => {
        this.handleExit(code, signal);
      },
      handleSpawnError: (error: Error): void => {
        this.handleSpawnError(error);
      },
    });
  }

  start(
    child: ChildProcessWithoutNullStreams | BridgeChildLike,
  ): Promise<AgentSession<TConfig>> {
    this.child = child;
    this.channel = createBridgeChannel(child, {
      graceMs: this.options.shutdownGraceMs,
      onSendError: (message: string): void => {
        // If stdin cannot accept a command, no backend reply can arrive. Close
        // the session so the initiating operation rejects instead of waiting.
        this.diagnostic({ kind: "send-failed", message });
        this.finishClose({ reason: "exit", code: null, signal: null });
        child.kill("SIGKILL");
      },
    });
    this.installExitHook(child);

    // A cold `uv run` may take time, but once this timer fires there is no
    // usable session handle. Startup rejection is distinct from `onClose`.
    this.readyTimer = setTimeout(() => {
      this.failStartup("Timed out waiting for bridge ready event.");
    }, this.options.readyTimeoutMs ?? 30_000);
    return this.readyDeferred.promise;
  }

  handleSpawnError(error: Error): void {
    if (this.readyInfo === null) {
      this.failStartup(error.message);
      return;
    }

    this.diagnostic({ kind: "send-failed", message: error.message });
  }

  handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    if (this.startupFailed) return;

    if (this.readyInfo === null) {
      // Exit before ready is startup failure, not a closed live session. The app
      // has not received a handle yet, so `onClose` must not fire.
      this.failStartup("Backend exited before ready event.", false);
      return;
    }

    const reason =
      this.closeRequested || this.shutdownSeen ? "shutdown" : "exit";
    this.finishClose({ reason, code, signal });
  }

  private session(): AgentSession<TConfig> {
    const ready = this.readyInfo;
    if (ready === null) throw new SessionStartError("Session is not ready.");

    return createSessionHandle(ready, {
      getConfig: (): Readonly<TConfig> | undefined => this.configState.current,
      run: (text: string): Promise<RunResult> => this.run(text),
      cancel: (): Promise<void> => this.cancel(),
      reset: (): Promise<void> => this.reset(),
      configure: (patch: Partial<TConfig>): Promise<Readonly<TConfig>> => {
        return this.configure(patch);
      },
      close: (): Promise<void> => this.close(),
    });
  }

  private handleEvent(event: BridgeEvent): void {
    if (event.type === "ready" && this.readyInfo === null) {
      this.resolveReady(event);
      if (this.startupFailed) return;
    }

    // Raw event observers see the strict protocol event before the session
    // reduces it into semantic callbacks.
    this.callAppHandler("onEvent", () => {
      this.options.onEvent?.(event);
    });

    // Command settlement happens before semantic processing so command-scoped
    // errors and terminal events cannot race a pending operation promise.
    if (event.type === "config") {
      this.configState.settleEvent(event, this.pendingCommands);
      return;
    }

    if (event.type === "reset" && event.config !== undefined) {
      if (this.configState.apply(event.config, true) === undefined) return;
    }

    this.settleCommandForEvent(event);
    this.reducer.process(event);
  }

  private resolveReady(event: BridgeEvent & { type: "ready" }): void {
    if (this.readyTimer !== null) {
      clearTimeout(this.readyTimer);
      this.readyTimer = null;
    }
    if (
      event.config !== undefined &&
      this.configState.apply(event.config, false) === undefined
    ) {
      return;
    }
    this.readyInfo = {
      version: event.version,
      package: event.package,
      metadata: event.metadata ?? {},
    };
    this.readyDeferred.resolve(this.session());
  }

  private run(text: string): Promise<RunResult> {
    if (this.closed !== null)
      return Promise.reject(new SessionClosedError(this.closed));
    if (this.activeRun !== null) {
      return Promise.reject(new SessionStateError("A run is already active."));
    }
    if (text.trim() === "") {
      return Promise.reject(new SessionStateError("Prompt must not be empty."));
    }

    const run = deferred<RunResult>();
    this.activeRun = run;
    this.send({ type: "prompt", text }, { kind: "prompt" });
    return run.promise;
  }

  private cancel(): Promise<void> {
    if (this.closed !== null)
      return Promise.reject(new SessionClosedError(this.closed));
    if (this.activeRun === null) return Promise.resolve();

    const waiter = deferred<void>();
    this.cancelWaiters.push(waiter);
    this.send({ type: "cancel" }, { kind: "cancel", reject: waiter.reject });
    return waiter.promise;
  }

  private reset(): Promise<void> {
    if (this.closed !== null)
      return Promise.reject(new SessionClosedError(this.closed));

    const waiter = deferred<void>();
    this.resetWaiters.push(waiter);
    this.send({ type: "reset" }, { kind: "reset", reject: waiter.reject });
    return waiter.promise;
  }

  private configure(patch: Partial<TConfig>): Promise<Readonly<TConfig>> {
    if (this.closed !== null) {
      return Promise.reject(new SessionClosedError(this.closed));
    }

    const patchDocument: Record<string, unknown> = { ...patch };
    const patchError = this.configState.validatePatch(patchDocument);
    if (patchError !== null) return Promise.reject(patchError);

    const waiter = deferred<Readonly<TConfig>>();
    this.send(
      { type: "configure", patch: patchDocument },
      {
        kind: "configure",
        resolveConfig: (config: Readonly<Record<string, unknown>>): void => {
          waiter.resolve(config as Readonly<TConfig>);
        },
        reject: waiter.reject,
      },
    );
    return waiter.promise;
  }

  private close(): Promise<void> {
    this.closeRequested = true;
    if (this.closeDeferred === null) {
      this.closeDeferred = deferred<void>();
    }

    if (this.closed !== null) {
      this.closeDeferred.resolve();
      return this.closeDeferred.promise;
    }

    // `createBridgeChannel.shutdown()` sends the command itself. We still track
    // it here so a command-scoped shutdown rejection has an owner if Python ever
    // emits one before the process exits.
    this.pendingCommands.push({ kind: "shutdown" });
    this.channel?.shutdown();
    return this.closeDeferred.promise;
  }

  private send(command: BridgeCommand, pending?: PendingCommand): boolean {
    if (this.channel === null || this.closed !== null) return false;

    if (pending !== undefined) {
      this.pendingCommands.push(pending);
    }

    const sent = this.channel.send(command);
    if (!sent && pending !== undefined) {
      // `onSendError` closes the session. Remove the optimistic pending entry so
      // command-error handling cannot later attribute an unrelated event to it.
      this.removePending(pending);
    }
    return sent;
  }

  private reducerCallbacks(): SessionReducerCallbacks {
    return createSessionReducerCallbacks(this.options, {
      activeRunIsIdle: (): boolean => this.activeRun === null,
      settleCancelWaiters: (): void => {
        this.settleCancelWaiters();
      },
      handleCommandError: (message: string): void => {
        this.handleCommandError(message);
      },
      diagnostic: (diagnostic: SessionDiagnostic): void => {
        this.diagnostic(diagnostic);
      },
      settleResetWaiters: (): void => {
        this.settleResetWaiters();
      },
      resolveRun: (result: RunResult): void => {
        this.resolveRun(result);
      },
      rejectRun: (error: Error): void => {
        this.rejectRun(error);
      },
      markShutdownSeen: (): void => {
        this.shutdownSeen = true;
      },
      sendApproval: (
        id: string,
        decision: { allowed: boolean; reason?: string },
      ): boolean =>
        this.send(
          {
            type: "approve",
            id,
            allowed: decision.allowed,
            reason: decision.reason,
          },
          { kind: "approve" },
        ),
    });
  }

  private handleCommandError(message: string): void {
    handlePendingCommandError(this.pendingCommands.shift(), message, {
      diagnostic: (diagnostic: SessionDiagnostic): void => {
        this.diagnostic(diagnostic);
      },
      rejectRun: (error: Error): void => {
        this.rejectRun(error);
      },
    });
  }

  private settleCommandForEvent(event: BridgeEvent): void {
    // Each command has one backend acknowledgement event. The queue may contain
    // multiple command types, so remove by kind rather than blindly shifting.
    if (event.type === "run_start") this.settleCommand("prompt");
    if (event.type === "approval_resolved") this.settleCommand("approve");
    if (event.type === "cancel_requested" || event.type === "cancel_ignored") {
      this.settleCommand("cancel");
    }
    if (event.type === "reset") this.settleCommand("reset");
    if (event.type === "shutdown") this.settleCommand("shutdown");
  }

  private settleCommand(kind: BridgeCommand["type"]): void {
    this.pendingCommands.take(kind);
  }

  private resolveRun(result: RunResult): void {
    const run = this.activeRun;
    this.activeRun = null;
    run?.resolve(result);
  }

  private rejectRun(error: Error): void {
    const run = this.activeRun;
    this.activeRun = null;
    run?.reject(error);
  }

  private settleCancelWaiters(): void {
    for (const waiter of this.cancelWaiters.splice(0)) {
      waiter.resolve();
    }
  }

  private settleResetWaiters(): void {
    for (const waiter of this.resetWaiters.splice(0)) {
      waiter.resolve();
    }
  }

  private closeForDecodeError(error: BridgeDecodeError): void {
    this.finishClose({
      reason: "protocol-error",
      error,
      code: null,
      signal: null,
    });
    this.child?.kill("SIGKILL");
  }

  private failProtocol(message: string, fields: readonly string[]): void {
    const error = new BridgeDecodeError(message, fields);
    this.diagnostic({ kind: "protocol", error, line: "" });

    if (this.readyInfo === null) {
      this.failStartup(message);
      return;
    }

    this.finishClose({
      reason: "protocol-error",
      error,
      code: null,
      signal: null,
    });
    this.child?.kill("SIGKILL");
  }

  private failStartup(message: string, kill: boolean = true): void {
    if (this.readyInfo !== null || this.startupFailed) return;

    this.startupFailed = true;
    if (this.readyTimer !== null) clearTimeout(this.readyTimer);
    this.removeExitHook();
    if (kill) this.child?.kill("SIGKILL");
    this.readyDeferred.reject(new SessionStartError(message));
  }

  private finishClose(close: SessionClose): void {
    if (this.closed !== null) return;

    this.closed = close;
    if (this.readyTimer !== null) clearTimeout(this.readyTimer);
    this.removeExitHook();

    // One terminal sweep owns the balance invariant: open text segments close,
    // open tools/agents become cancelled, and pending approval policies abort.
    this.reducer.sweepTerminal();
    this.reducer.dispose();

    const closedError = new SessionClosedError(close);
    this.rejectRun(closedError);
    this.pendingCommands.rejectAll(closedError);
    for (const waiter of this.cancelWaiters.splice(0))
      waiter.reject(closedError);
    for (const waiter of this.resetWaiters.splice(0))
      waiter.reject(closedError);

    this.closeDeferred?.resolve();
    if (this.readyInfo !== null) {
      this.callAppHandler("onClose", () => {
        this.options.onClose?.(close);
      });
    }
  }

  private installExitHook(child: BridgeChildLike): void {
    this.exitHook = (): void => {
      child.kill("SIGKILL");
    };
    // Node's process exit event is the last chance to avoid orphaning the local
    // Python backend when the app terminates without calling close().
    process.on("exit", this.exitHook);
  }

  private removeExitHook(): void {
    if (this.exitHook === null) return;

    process.removeListener("exit", this.exitHook);
    this.exitHook = null;
  }

  private removePending(pending: PendingCommand): void {
    this.pendingCommands.remove(pending);
  }

  private diagnostic(diagnostic: SessionDiagnostic): void {
    if (this.options.onDiagnostic === undefined) {
      console.error("[session]", diagnostic);
      return;
    }

    this.callAppHandler("onDiagnostic", () => {
      this.options.onDiagnostic?.(diagnostic);
    });
  }

  private callAppHandler(handler: string, call: () => void): void {
    try {
      call();
    } catch (error) {
      if (handler === "onDiagnostic") {
        // Avoid recursive diagnostic failure loops. The default sink is still
        // visible, but it cannot call back into app code again.
        console.error("[session]", { kind: "handler-error", handler, error });
        return;
      }

      this.diagnostic({ kind: "handler-error", handler, error });
    }
  }
}
