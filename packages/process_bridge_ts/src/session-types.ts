import type { BackendSpec } from "./backend-spec.ts";
import type { BridgeDecodeError } from "./decoders.ts";
import type {
  ApprovalRequestPayload,
  BridgeEvent,
  ConfigErrorCode,
  ToolErrorPayload,
} from "./protocol.ts";
import type { TextDelivery } from "./text-stream.ts";

/** App-facing alias for the configure failure code emitted by the bridge. */
export type ConfigureErrorCode = ConfigErrorCode;

/**
 * One piece of streamed assistant or reasoning text.
 *
 * Chunks of one contiguous stream share a `segment`; a new segment opens when
 * a tool call or the other stream kind interrupts. Segment ids are suitable as
 * stable row keys in a terminal or chat UI.
 */
export type TextChunk = {
  /** New text since the previous chunk of this segment. May be empty on `done`. */
  delta: string;

  /** Full accumulated text of the segment, including this delta. */
  text: string;

  /** Monotonic segment counter, unique per session. */
  segment: number;

  /**
   * True on the segment's final chunk, exactly once per opened segment.
   *
   * On the run's last assistant segment, `text` is reconciled against the
   * authoritative `run_complete.final_output`. If `final_output` is non-empty
   * with no open assistant segment, the session synthesizes a one-chunk segment.
   */
  done: boolean;
};

/** Identity of one tool call, stable across its phases. */
export type ToolCallInfo = {
  /**
   * Wire `tool_call_id` when it is non-blank; a synthesized stable id only when
   * the backend sends an empty id. Approval linkage uses the wire id:
   * `request.tool_call_id === callId` for every non-blank tool call id.
   */
  callId: string;

  /** Tool name visible to the model and app UI. */
  tool: string;

  /** Agent task that issued the tool call. */
  agent: string;

  /** Task id that owns the tool call. */
  taskId: string;

  /** Parsed argument object from the backend, or the raw payload when needed. */
  arguments: unknown;

  /** True when this is the framework planning tool. */
  isPlan: boolean;

  /** True when this tool call delegated work to another agent. */
  isDelegation: boolean;
};

/**
 * Tool lifecycle, balanced by construction.
 *
 * The session synthesizes `cancelled`, never a fake `end`, for calls left open
 * by cancellation, reset, run failure, close, or backend death. Plan-tool calls
 * are not suppressed: they appear with `isPlan: true` and may also produce
 * `onPlan`; presentation is the app's choice.
 */
export type ToolActivity =
  | { phase: "start"; call: ToolCallInfo }
  | {
      phase: "end";
      call: ToolCallInfo;
      ok: boolean;
      result: unknown;
      error: ToolErrorPayload | null;
    }
  | { phase: "cancelled"; call: ToolCallInfo };

/** Identity of one agent task, root or sub-agent, stable across phases. */
export type AgentInfo = {
  agent: string;
  taskId: string;
  parentTaskId: string | null;
  isRoot: boolean;
};

/** Agent task lifecycle. Balanced like `ToolActivity`. */
export type AgentActivity =
  | { phase: "start"; info: AgentInfo }
  | { phase: "end"; info: AgentInfo; finalPreview: string | null }
  | { phase: "cancelled"; info: AgentInfo };

/** Normalized plan-step status; `PlanUpdate.rawStatus` preserves the wire string. */
export type PlanStepStatus = "pending" | "active" | "completed" | "blocked";

/** Plan snapshot from the AgentLane plan tool. */
export type PlanUpdate = {
  agent: string;
  taskId: string;
  explanation: string | null;
  steps: Array<{ text: string; status: PlanStepStatus; rawStatus: string }>;
};

/**
 * Approval awaiting the app's decision.
 *
 * The wire payload stays nested: `request.reason` is the payload's own field,
 * while top-level `reason` is the event-level reason the permission gate fired.
 */
export type ApprovalRequest = {
  /** Validated wire payload with tool name, operation, cwd, path, command, and metadata. */
  request: ApprovalRequestPayload;

  /** Why the permission gate fired, when the backend provided one. */
  reason: string | null;

  /**
   * Aborts if the request is resolved before the policy returns.
   *
   * Causes include run cancellation, reset, run failure, session close, backend
   * death, or a backend-side resolution arriving first. Decisions after abort
   * are discarded and never sent to Python.
   */
  signal: AbortSignal;
};

/** `true`/`false` for the simple case; object form attaches a denial reason. */
export type ApprovalDecision = boolean | { allowed: boolean; reason?: string };

/**
 * The app's approval policy.
 *
 * Called once per request and may run concurrently when parallel tool calls
 * gate at once. Throwing denies with reason "Approval policy failed." and emits
 * a handler-error diagnostic. No policy configured means deny-all with a clear
 * explanatory reason, so the run fails visibly instead of hanging.
 */
export type ApprovalPolicy = (
  request: ApprovalRequest,
) => ApprovalDecision | Promise<ApprovalDecision>;

/**
 * Resolved approval state confirmed by `approval_resolved`.
 *
 * This is the source of truth. It fires for app decisions and backend-initiated
 * resolutions such as Python policy denials or force denials during shutdown.
 */
export type ApprovalResolution = {
  request: ApprovalRequestPayload;
  allowed: boolean;
  reason: string | null;

  /** False when the backend resolved it without the app's decision. */
  decidedByApp: boolean;
};

/**
 * Best-effort observability for the session layer.
 *
 * Ignoring diagnostics never affects event delivery or promise settlement.
 * Default sink: `console.error` prefixed with `[session]`. A `protocol`
 * diagnostic is always followed by teardown because strict decode failure means
 * Python and TypeScript are out of sync.
 */
export type SessionDiagnostic =
  | { kind: "protocol"; error: BridgeDecodeError; line: string }
  | { kind: "stderr"; line: string }
  | { kind: "send-failed"; message: string }
  | { kind: "command-rejected"; message: string }
  | { kind: "handler-error"; handler: string; error: unknown };

/**
 * Why a started session stopped.
 *
 * `onClose` fires exactly once for these reasons. Startup failures reject
 * `createAgentSession` instead and never call `onClose`.
 */
export type SessionClose =
  | { reason: "shutdown"; code: number | null; signal: NodeJS.Signals | null }
  | { reason: "exit"; code: number | null; signal: NodeJS.Signals | null }
  | {
      reason: "protocol-error";
      error: BridgeDecodeError;
      code: number | null;
      signal: NodeJS.Signals | null;
    };

/** Backend identity from the ready event. */
export type ReadyInfo = {
  version: string;
  package: string;

  /** App-defined ready metadata; `{}` when the backend omitted it. */
  metadata: Record<string, unknown>;
};

/**
 * Outcome of one run.
 *
 * Cancellation is a normal app-requested outcome, so it resolves rather than
 * rejects. Provider failures and backend run errors reject with `RunError`.
 */
export type RunResult =
  | {
      status: "completed";
      finalOutput: string;
      turnCount: number;
      responseCount: number;
    }
  | { status: "cancelled" };

/** Startup failed: spawn error, exit before ready, bad backend spec, or ready timeout. */
export class SessionStartError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SessionStartError";
  }
}

/** The backend rejected or failed the active run. */
export class RunError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RunError";
  }
}

/** The backend stopped while an operation was in flight. */
export class SessionClosedError extends Error {
  readonly close: SessionClose;

  constructor(close: SessionClose) {
    super(`Agent session closed: ${close.reason}`);
    this.name = "SessionClosedError";
    this.close = close;
  }
}

/** Local API misuse checked before touching the wire. */
export class SessionStateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SessionStateError";
  }
}

/**
 * Backend-reported runtime configuration failure.
 *
 * `rejected` carries the Python store's user-presentable message. `invalid`,
 * `unsupported`, and `internal` are transport-level failures; `internal` uses a
 * fixed non-leaking message while Python logs the traceback to stderr.
 */
export class ConfigureError extends Error {
  readonly code: ConfigureErrorCode;

  constructor(code: ConfigureErrorCode, message: string) {
    super(message);
    this.name = "ConfigureError";
    this.code = code;
  }
}

/**
 * Raw protocol tap used by session options.
 *
 * This intentionally exposes `BridgeEvent`: once an app subscribes here it has
 * chosen to look below the session layer.
 */
export type RawEventHandler = (event: BridgeEvent) => void;

/**
 * The app's declared behavior for one local AgentLane session.
 *
 * Every handler is optional. Handlers fire synchronously in wire order; the
 * session catches handler exceptions, reports them as handler-error
 * diagnostics, and continues. An app bug must not corrupt correlation state or
 * strand a pending operation promise.
 */
export type AgentSessionOptions<
  TConfig extends object = Record<string, unknown>,
> = {
  /** Backend process to spawn. The only required field. */
  backend: BackendSpec;

  /** Approval policy. Default: deny-all with an explanatory reason. */
  approvals?: ApprovalPolicy;

  /** Balanced agent and sub-agent task lifecycle. */
  onAgentActivity?: (activity: AgentActivity) => void;

  /** Every approval resolution, app-decided or backend-initiated. */
  onApprovalResolved?: (resolution: ApprovalResolution) => void;

  /** Assistant text, segmented and coalesced by default. */
  onAssistantText?: (chunk: TextChunk) => void;

  /** Fired exactly once when a started session stops, for any reason. */
  onClose?: (close: SessionClose) => void;

  /**
   * Runtime decoder for backend-announced config documents.
   *
   * This is the app's zod/io-ts seam for its own Python store. Throwing means
   * lockstep drift between the app frontend and backend, so the session closes
   * like a protocol error instead of exposing a partially trusted document.
   * When omitted, config documents are exposed under an unchecked cast.
   */
  decodeConfig?: (raw: Record<string, unknown>) => TConfig;

  /** Best-effort diagnostics. Default: `console.error` with `[session]`. */
  onDiagnostic?: (diagnostic: SessionDiagnostic) => void;

  /**
   * Fires after startup whenever the backend re-announces authoritative config.
   *
   * The initial document is intentionally not delivered here; read
   * `session.config` after `createAgentSession` resolves so UI setup can bind
   * catalog display data and initial selection at one call site.
   */
  onConfigChanged?: (config: Readonly<TConfig>) => void;

  /**
   * Raw protocol tap: every strictly decoded event, at receipt, in wire order,
   * before semantic processing.
   *
   * This is the deliberate window below the session line. `BridgeEvent` is the
   * closed protocol union, so exhaustive switches compile. Handoff spans, LLM
   * spans, state snapshots, and provider passthrough live here by design.
   */
  onEvent?: (event: BridgeEvent) => void;

  /** Plan snapshots with normalized step statuses. */
  onPlan?: (plan: PlanUpdate) => void;

  /** Reasoning text, with the same segment semantics as assistant text. */
  onReasoningText?: (chunk: TextChunk) => void;

  /** Balanced tool-call lifecycle, correlated by the session. */
  onToolActivity?: (activity: ToolActivity) => void;

  /** Max wait for ready before startup rejects. Defaults to `30000` for uv cold starts. */
  readyTimeoutMs?: number;

  /** Cooperative shutdown grace before SIGKILL. Defaults to `1500`. */
  shutdownGraceMs?: number;

  /** Text delivery mode. Defaults to `"coalesced"`. */
  textDelivery?: TextDelivery;
};

/**
 * Handle to a running local AgentLane session.
 *
 * One conversation, one run at a time. After `onClose`, the handle is dead; the
 * options object is stateless and may be passed to `createAgentSession` again
 * for a fresh backend.
 */
export type AgentSession<
  TConfig extends object = Record<string, unknown>,
  TConfigPatch extends object = Partial<TConfig>,
> = {
  /** Backend identity captured from `ready`. */
  ready: ReadyInfo;

  /**
   * Latest authoritative runtime config, or `undefined` when the backend has no
   * config store. This cache updates only from backend announcements.
   */
  readonly config: Readonly<TConfig> | undefined;

  /**
   * Start a run.
   *
   * Resolves after the terminal event's full handler cascade has run. Rejects
   * `SessionStateError` on local misuse, `RunError` on backend run failure, and
   * `SessionClosedError` on backend death. Never throws synchronously.
   */
  run: (text: string) => Promise<RunResult>;

  /**
   * Cancel the active run.
   *
   * Resolves once the run reaches any terminal state; natural completion
   * winning the race is fine. Resolves immediately when idle, and
   * `cancel_ignored` is absorbed.
   */
  cancel: () => Promise<void>;

  /**
   * Reset conversation state.
   *
   * Any active run is cancelled first, resolving that run as `cancelled`, then
   * this resolves on the backend's reset confirmation.
   */
  reset: () => Promise<void>;

  /**
   * Apply a desired-state patch to the Python runtime config store.
   *
   * `TConfigPatch` is intentionally separate from `TConfig`: apps may expose a
   * narrow write shape while the backend announces a richer truth document.
   *
   * The backend owns validation and normalization, then settles with the full
   * applied document. Resolves with that document on success. Rejects with
   * `ConfigureError` on backend-reported failure after any included truth
   * snapshot has already refreshed `session.config`.
   */
  configure: (patch: TConfigPatch) => Promise<Readonly<TConfig>>;

  /**
   * Graceful shutdown.
   *
   * Sends the protocol shutdown command, waits for bounded grace, escalates to
   * SIGKILL when needed, and removes the process exit hook. Idempotent and
   * never rejects.
   */
  close: () => Promise<void>;
};
