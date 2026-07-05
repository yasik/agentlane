import type { BridgeDecodeError } from "./decoders.ts";
import type { BridgeEvent } from "./protocol.ts";
import type { SessionReducerCallbacks } from "./session-reducer.ts";
import type {
  AgentSessionOptions,
  RunResult,
  SessionDiagnostic,
} from "./session-types.ts";
import { RunError } from "./session-types.ts";

/**
 * Controller operations needed by the process wiring adapter.
 *
 * The adapter receives raw child-process callbacks; the controller still owns
 * session state and decides what closing or diagnostics mean.
 */
type ProcessHandlers = {
  /** Record a non-fatal process or protocol observation for app diagnostics. */
  diagnostic: (diagnostic: SessionDiagnostic) => void;

  /** Tear down the session after stdout produces an invalid protocol line. */
  closeForDecodeError: (error: BridgeDecodeError) => void;

  /** Hand a strictly decoded bridge event to the session controller. */
  handleEvent: (event: BridgeEvent) => void;

  /** Let the controller settle close semantics after child-process exit. */
  handleExit: (code: number | null, signal: NodeJS.Signals | null) => void;

  /** Reject startup or close an already-started session after spawn failure. */
  handleSpawnError: (error: Error) => void;
};

/**
 * Build callbacks for `wireBridgeProcess`.
 *
 * This adapter is intentionally thin: process events are normalized into
 * controller methods, while protocol decode failure remains the only stdout
 * read failure that forces immediate session teardown.
 */
export function createSessionProcessCallbacks(handlers: ProcessHandlers): {
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
      handlers.diagnostic({ kind: "protocol", error, line });
      handlers.closeForDecodeError(error);
    },
    onEvent: handlers.handleEvent,
    onExit: handlers.handleExit,
    onSpawnError: handlers.handleSpawnError,
    onStderr: (line: string): void => {
      handlers.diagnostic({ kind: "stderr", line });
    },
  };
}

type ReducerHandlers = {
  /** True when no run promise is waiting for a terminal event. */
  activeRunIsIdle: () => boolean;

  /** Resolve every cancel() waiter once the run has reached terminal state. */
  settleCancelWaiters: () => void;

  /** Route a backend command error through command FIFO ownership rules. */
  handleCommandError: (message: string) => void;

  /** Emit reducer diagnostics without letting them change promise settlement. */
  diagnostic: (diagnostic: SessionDiagnostic) => void;

  /** Resolve reset() waiters after the reducer observes reset completion. */
  settleResetWaiters: () => void;

  /** Resolve the active run promise with the reducer's terminal result. */
  resolveRun: (result: RunResult) => void;

  /** Reject the active run promise with the reducer's terminal error. */
  rejectRun: (error: Error) => void;

  /** Remember that the backend sent the protocol shutdown event before exit. */
  markShutdownSeen: () => void;

  /** Send an approval decision back to Python while the request is still live. */
  sendApproval: (
    id: string,
    decision: { allowed: boolean; reason?: string },
  ) => boolean;
};

/**
 * Build callbacks for semantic event reduction.
 *
 * `SessionReducer` owns balanced UI events; the controller owns public promise
 * settlement. This adapter is the contract between the two, translating
 * reducer terminal callbacks into operation outcomes without letting reducer
 * code see child-process or command-queue state.
 */
export function createSessionReducerCallbacks<
  TConfig extends Record<string, unknown>,
>(
  options: AgentSessionOptions<TConfig>,
  handlers: ReducerHandlers,
): SessionReducerCallbacks {
  return {
    approvals: options.approvals,
    onAgentActivity: options.onAgentActivity,
    onApprovalResolved: options.onApprovalResolved,
    onAssistantText: options.onAssistantText,
    onCancelSettled: (): void => {
      if (handlers.activeRunIsIdle()) handlers.settleCancelWaiters();
    },
    onCommandError: handlers.handleCommandError,
    onDiagnostic: handlers.diagnostic,
    onPlan: options.onPlan,
    onReasoningText: options.onReasoningText,
    onReset: handlers.settleResetWaiters,
    onRunCancelled: (): void => {
      // Cancellation is a normal operation result, not an exception. Resolve
      // the active run and then unblock any cancel() callers waiting for the
      // terminal run event.
      handlers.resolveRun({ status: "cancelled" });
      handlers.settleCancelWaiters();
    },
    onRunCompleted: (result: RunResult): void => {
      handlers.resolveRun(result);
      handlers.settleCancelWaiters();
    },
    onRunError: (message: string): void => {
      // Run-scoped errors terminate the active run; they are not command
      // rejections and should not be attributed through the command FIFO.
      handlers.rejectRun(new RunError(message));
      handlers.settleCancelWaiters();
    },
    onRunStarted: (): void => undefined,
    onShutdown: handlers.markShutdownSeen,
    onToolActivity: options.onToolActivity,
    sendApproval: handlers.sendApproval,
    textDelivery: options.textDelivery,
  };
}
