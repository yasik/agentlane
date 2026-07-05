import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { resolveBackendSpec } from "./backend-spec.ts";
import type { BridgeChildLike } from "./channel.ts";
import { createBridgeChannel } from "./channel.ts";
import type { BridgeDecodeError } from "./decoders.ts";
import type { BridgeProcessOptions } from "./process.ts";
import { spawnBridgeProcess } from "./process.ts";
import type { BridgeCommand, BridgeEvent } from "./protocol.ts";
import {
  SessionReducer,
  type SessionReducerCallbacks,
} from "./session-reducer.ts";
import {
  type AgentSession,
  type AgentSessionOptions,
  type ReadyInfo,
  RunError,
  type RunResult,
  type SessionClose,
  SessionClosedError,
  type SessionDiagnostic,
  SessionStartError,
  SessionStateError,
} from "./session-types.ts";

/**
 * Command waiting for a backend acknowledgement or command-scoped error.
 *
 * The backend handles commands serially, so a command-scoped error belongs to
 * the oldest unsettled command. This keeps rejected prompts from hanging
 * `run()` without adding event-specific heuristics.
 */
type PendingCommand = {
  kind: BridgeCommand["type"];
  reject?: (error: Error) => void;
};

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
};

/**
 * Spawn the backend, wire stdio, and resolve once `ready` arrives.
 *
 * Startup failures reject with `SessionStartError` before a live session is
 * exposed. After startup, all operation promises settle through the session
 * lifecycle; app code never needs to observe the raw child process.
 */
export function createAgentSession(
  options: AgentSessionOptions,
): Promise<AgentSession> {
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

  const controller = new AgentSessionController(options);
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
export class AgentSessionController {
  private readonly options: AgentSessionOptions;
  private readonly readyDeferred = deferred<AgentSession>();
  private readonly pendingCommands: PendingCommand[] = [];
  private readonly cancelWaiters: Array<Deferred<void>> = [];
  private readonly resetWaiters: Array<Deferred<void>> = [];
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

  constructor(options: AgentSessionOptions) {
    this.options = options;
    this.reducer = new SessionReducer(this.reducerCallbacks());
  }

  processCallbacks(): {
    onDecodeError: (error: BridgeDecodeError, line: string) => void;
    onEvent: (event: BridgeEvent) => void;
    onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
    onSpawnError: (error: Error) => void;
    onStderr: (line: string) => void;
  } {
    return {
      onDecodeError: (error: BridgeDecodeError, line: string): void => {
        // Decode failure is fatal under the strict-companion contract. A lost
        // terminal event can strand app promises, so we tear down immediately.
        this.diagnostic({ kind: "protocol", error, line });
        this.finishClose({
          reason: "protocol-error",
          error,
          code: null,
          signal: null,
        });
        this.child?.kill("SIGKILL");
      },
      onEvent: (event: BridgeEvent): void => {
        this.handleEvent(event);
      },
      onExit: (code: number | null, signal: NodeJS.Signals | null): void => {
        this.handleExit(code, signal);
      },
      onSpawnError: (error: Error): void => {
        this.handleSpawnError(error);
      },
      onStderr: (line: string): void => {
        this.diagnostic({ kind: "stderr", line });
      },
    };
  }

  start(
    child: ChildProcessWithoutNullStreams | BridgeChildLike,
  ): Promise<AgentSession> {
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

  private session(): AgentSession {
    const ready = this.readyInfo;
    if (ready === null) throw new SessionStartError("Session is not ready.");

    return {
      ready,
      run: (text: string): Promise<RunResult> => this.run(text),
      cancel: (): Promise<void> => this.cancel(),
      reset: (): Promise<void> => this.reset(),
      close: (): Promise<void> => this.close(),
    };
  }

  private handleEvent(event: BridgeEvent): void {
    if (event.type === "ready" && this.readyInfo === null) {
      this.resolveReady(event);
    }

    // Raw event observers see the strict protocol event before the session
    // reduces it into semantic callbacks.
    this.callAppHandler("onEvent", () => {
      this.options.onEvent?.(event);
    });

    // Command settlement happens before semantic processing so command-scoped
    // errors and terminal events cannot race a pending operation promise.
    this.settleCommandForEvent(event);
    this.reducer.process(event);
  }

  private resolveReady(event: BridgeEvent & { type: "ready" }): void {
    if (this.readyTimer !== null) {
      clearTimeout(this.readyTimer);
      this.readyTimer = null;
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
    return {
      approvals: this.options.approvals,
      onAgentActivity: this.options.onAgentActivity,
      onApprovalResolved: this.options.onApprovalResolved,
      onAssistantText: this.options.onAssistantText,
      onCancelSettled: (): void => {
        if (this.activeRun === null) this.settleCancelWaiters();
      },
      onCommandError: (message: string): void => {
        this.handleCommandError(message);
      },
      onDiagnostic: (diagnostic: SessionDiagnostic): void => {
        this.diagnostic(diagnostic);
      },
      onPlan: this.options.onPlan,
      onReasoningText: this.options.onReasoningText,
      onReset: (): void => {
        this.settleResetWaiters();
      },
      onRunCancelled: (): void => {
        this.resolveRun({ status: "cancelled" });
        this.settleCancelWaiters();
      },
      onRunCompleted: (result: RunResult): void => {
        this.resolveRun(result);
        this.settleCancelWaiters();
      },
      onRunError: (message: string): void => {
        this.rejectRun(new RunError(message));
        this.settleCancelWaiters();
      },
      onRunStarted: (): void => undefined,
      onShutdown: (): void => {
        this.shutdownSeen = true;
      },
      onToolActivity: this.options.onToolActivity,
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
      textDelivery: this.options.textDelivery,
    };
  }

  private handleCommandError(message: string): void {
    const command = this.pendingCommands.shift();
    if (command === undefined) {
      this.diagnostic({ kind: "command-rejected", message });
      return;
    }

    if (command.kind === "prompt") {
      // Prompt rejection is the only command error that directly belongs to a
      // public run promise. Local prechecks catch the common misuse cases first.
      this.rejectRun(new RunError(message));
      return;
    }

    if (command.kind === "cancel" || command.kind === "reset") {
      command.reject?.(new RunError(message));
      return;
    }

    this.diagnostic({ kind: "command-rejected", message });
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
    const index = this.pendingCommands.findIndex(
      (command: PendingCommand): boolean => command.kind === kind,
    );
    if (index >= 0) this.pendingCommands.splice(index, 1);
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
    const index = this.pendingCommands.indexOf(pending);
    if (index >= 0) this.pendingCommands.splice(index, 1);
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

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}
