import type { BridgeEvent } from "./protocol.ts";
import {
  isKnownModelEvent,
  isNativeEvent,
  type NativeApprovalRecord,
  type NativePayload,
  type NativeRunEvent,
} from "./protocol-native.ts";
import type {
  AgentActivity,
  AgentInfo,
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalResolution,
  PlanStepStatus,
  PlanUpdate,
  RunResult,
  SessionDiagnostic,
  TextChunk,
  ToolActivity,
  ToolCallInfo,
} from "./session-types.ts";
import {
  type TextDelivery,
  type TextStreamKind,
  TextStreamTracker,
} from "./text-stream.ts";

type PendingApproval = {
  controller: AbortController;
  decidedByApp: boolean;
};

/**
 * Boundary between raw bridge events and session-controller side effects.
 *
 * The reducer owns semantic correlation, but the controller owns operation
 * promises and child-process state. Keeping this interface explicit prevents
 * app callback failures from leaking into transport lifecycle code.
 */
export type SessionReducerCallbacks = {
  approvals?: ApprovalPolicy;
  onAgentActivity?: (activity: AgentActivity) => void;
  onApprovalResolved?: (resolution: ApprovalResolution) => void;
  onAssistantText?: (chunk: TextChunk) => void;
  onCancelSettled: () => void;
  onCommandError: (message: string) => void;
  onDiagnostic: (diagnostic: SessionDiagnostic) => void;
  onPlan?: (plan: PlanUpdate) => void;
  onReasoningText?: (chunk: TextChunk) => void;
  onReset: () => void;
  onRunCancelled: () => void;
  onRunCompleted: (result: RunResult) => void;
  onRunError: (message: string) => void;
  onRunStarted: () => void;
  onShutdown: () => void;
  onToolActivity?: (activity: ToolActivity) => void;
  sendApproval: (
    id: string,
    decision: { allowed: boolean; reason?: string },
  ) => boolean;
  textDelivery?: TextDelivery;
};

/**
 * Converts raw bridge events into balanced app-level session callbacks.
 *
 * This class is intentionally stateful: it tracks open text segments, tool
 * calls, agent tasks, and approvals so every terminal path can close or cancel
 * visible UI rows exactly once.
 */
export class SessionReducer {
  private readonly callbacks: SessionReducerCallbacks;
  private readonly text: TextStreamTracker;
  private readonly openTools = new Map<string, ToolCallInfo>();
  private readonly openAgents = new Map<string, AgentInfo>();
  private readonly pendingApprovals = new Map<string, PendingApproval>();

  constructor(callbacks: SessionReducerCallbacks) {
    this.callbacks = callbacks;
    this.text = new TextStreamTracker(
      {
        onAssistantText: (chunk: TextChunk): void => {
          this.callHandler("onAssistantText", () =>
            this.callbacks.onAssistantText?.(chunk),
          );
        },
        onReasoningText: (chunk: TextChunk): void => {
          this.callHandler("onReasoningText", () =>
            this.callbacks.onReasoningText?.(chunk),
          );
        },
      },
      callbacks.textDelivery,
    );
  }

  process(event: BridgeEvent): void {
    switch (event.type) {
      case "ready":
        return;
      case "config":
        // Config events are consumed by the controller because they settle
        // `configure()` promises and update the session cache. They have no
        // transcript lifecycle effect for the reducer.
        return;
      case "run_start":
        this.callbacks.onRunStarted();
        return;
      case "run_event":
        this.processNative(event.event);
        return;
      case "run_complete":
        // Completion is the only terminal path with an authoritative final
        // assistant output. Reconcile text before resolving the run.
        this.completeRun(event.final_output);
        this.callbacks.onRunCompleted({
          status: "completed",
          finalOutput: event.final_output,
          turnCount: event.turn_count,
          responseCount: event.response_count,
        });
        return;
      case "run_cancelled":
        this.sweepTerminal();
        this.callbacks.onRunCancelled();
        return;
      case "cancel_requested":
      case "cancel_ignored":
        this.flushText();
        this.callbacks.onCancelSettled();
        return;
      case "reset":
        this.sweepTerminal();
        this.callbacks.onReset();
        return;
      case "shutdown":
        this.sweepTerminal();
        this.callbacks.onShutdown();
        return;
      case "error":
        if (event.scope === "command") {
          // Command errors are attributed by the controller's command FIFO; they
          // are not run-terminal events by themselves.
          this.callbacks.onCommandError(event.message);
          return;
        }

        this.sweepTerminal();
        this.callbacks.onRunError(event.message);
        return;
      default:
        // Known-but-not-semantic events such as provider, LLM, handoff, and
        // state snapshots still act as boundaries for coalesced text delivery.
        this.flushText();
        return;
    }
  }

  private processNative(event: NativeRunEvent): void {
    if (isNativeEvent(event, "model_stream")) {
      const model = event.payload.event;
      if (isKnownModelEvent(model)) {
        if (model.kind === "text_delta" && typeof model.text === "string") {
          this.pushText("assistant", model.text);
          return;
        }
        if (model.kind === "reasoning") {
          const text = model.text ?? model.reasoning;
          if (typeof text === "string") this.pushText("reasoning", text);
          return;
        }
      }
      this.flushText();
      return;
    }
    if (isNativeEvent(event, "tool_start")) {
      this.toolStarted(event.payload);
      return;
    }

    if (isNativeEvent(event, "tool_end")) {
      this.toolEnded(event.payload);
      return;
    }

    if (isNativeEvent(event, "agent_start")) {
      this.agentStarted(event.payload);
      return;
    }

    if (isNativeEvent(event, "agent_end")) {
      this.agentEnded(event.payload);
      return;
    }

    if (isNativeEvent(event, "plan_updated")) {
      this.planUpdated(event.payload);
      return;
    }

    if (isNativeEvent(event, "tool_approval")) {
      const record = event.payload.event.record;
      if (record.status === "pending") {
        this.approvalRequested(record);
        return;
      }
      this.approvalResolved(record);
      return;
    }
    this.flushText();
  }

  /** Close all open semantic state for a terminal run/session path. */
  sweepTerminal(): void {
    this.text.complete();
    this.cancelOpenTools();
    this.cancelOpenAgents();
    this.abortApprovals();
  }

  dispose(): void {
    this.text.dispose();
    this.abortApprovals();
  }

  private pushText(kind: TextStreamKind, delta: string): void {
    this.text.push(kind, delta);
  }

  private flushText(): void {
    this.text.flush();
  }

  private interruptText(): void {
    this.text.interrupt();
  }

  private toolStarted(event: NativePayload<"tool_start">): void {
    // Tool rows interrupt streamed prose in the UI; close the current text
    // segment before emitting the tool start.
    this.interruptText();
    const call = this.toolCallInfo(event);
    this.openTools.set(call.callId, call);
    this.emitTool({ phase: "start", call });
  }

  private toolEnded(event: NativePayload<"tool_end">): void {
    this.interruptText();
    const call =
      this.openTools.get(event.tool_call.id) ?? this.toolCallInfo(event);
    this.openTools.delete(call.callId);
    this.emitTool({
      phase: "end",
      call,
      ok: event.ok,
      result: event.result,
      error: event.error,
    });
  }

  private toolCallInfo(
    event: NativePayload<"tool_start" | "tool_end">,
  ): ToolCallInfo {
    return {
      callId: event.tool_call.id,
      tool: event.tool_call.function.name,
      agent: event.task_name,
      taskId: event.task_id,
      arguments: parseToolArguments(event.tool_call.function.arguments),
      isPlan: event.tool_call.function.name === "write_plan",
      isDelegation: event.is_delegation,
    };
  }

  private agentStarted(event: NativePayload<"agent_start">): void {
    // Agent lifecycle is a structural boundary, but it does not invalidate the
    // active text stream; flush pending text without marking it done.
    this.flushText();
    const info = this.agentInfo(event);
    this.openAgents.set(info.taskId, info);
    this.emitAgent({ phase: "start", info });
  }

  private agentEnded(event: NativePayload<"agent_end">): void {
    this.flushText();
    const info = this.openAgents.get(event.task_id) ?? this.agentInfo(event);
    this.openAgents.delete(info.taskId);
    this.emitAgent({
      phase: "end",
      info,
      finalOutput: event.result?.final_output,
    });
  }

  private agentInfo(
    event: NativePayload<"agent_start" | "agent_end">,
  ): AgentInfo {
    return {
      agent: event.task_name,
      taskId: event.task_id,
      parentTaskId: event.parent_task_id,
      isRoot: event.is_root,
    };
  }

  private planUpdated(event: NativePayload<"plan_updated">): void {
    this.flushText();
    this.callHandler("onPlan", () =>
      this.callbacks.onPlan?.({
        agent: event.task_name,
        taskId: event.task_id,
        explanation: event.explanation,
        steps: event.plan.map((step) => ({
          text: step.step,
          status: normalizePlanStatus(step.status),
          rawStatus: step.status,
        })),
      }),
    );
  }

  private approvalRequested(event: NativeApprovalRecord): void {
    this.flushText();
    const controller = new AbortController();
    this.pendingApprovals.set(event.request_id, {
      controller,
      decidedByApp: false,
    });
    const policy = this.callbacks.approvals ?? denyAllApproval;

    // Approval policy code is app-owned and may be async. Keep it outside the
    // event dispatch stack so a slow modal cannot block protocol processing.
    Promise.resolve()
      .then((): ApprovalDecision | Promise<ApprovalDecision> =>
        policy({
          request: event.request,
          reason: event.approval_required_decision.reason,
          signal: controller.signal,
        }),
      )
      .then((decision: ApprovalDecision): void => {
        this.sendApprovalDecision(event.request_id, decision);
      })
      .catch((error: unknown): void => {
        this.callbacks.onDiagnostic({
          kind: "handler-error",
          handler: "approvals",
          error,
        });
        this.sendApprovalDecision(event.request_id, {
          allowed: false,
          reason: "Approval policy failed.",
        });
      });
  }

  private sendApprovalDecision(id: string, decision: ApprovalDecision): void {
    const pending = this.pendingApprovals.get(id);
    if (pending === undefined || pending.controller.signal.aborted) return;

    const normalized = normalizeApprovalDecision(decision);
    if (this.callbacks.sendApproval(id, normalized)) {
      // The native resolved approval record is the source of truth. This flag tells
      // the app whether its policy produced the resolution Python confirmed.
      pending.decidedByApp = true;
    }
  }

  private approvalResolved(event: NativeApprovalRecord): void {
    this.flushText();
    const pending = this.pendingApprovals.get(event.request_id);
    this.pendingApprovals.delete(event.request_id);
    // Abort even after a normal app decision so any UI waiting on the signal can
    // dismiss once Python confirms the approval is no longer pending.
    pending?.controller.abort();
    this.callHandler("onApprovalResolved", () =>
      this.callbacks.onApprovalResolved?.({
        request: event.request,
        allowed: event.final_decision?.outcome === "allow",
        reason: event.final_decision?.reason ?? null,
        decidedByApp: pending?.decidedByApp ?? false,
      }),
    );
  }

  private completeRun(finalOutput: unknown): void {
    // Structured results belong to the run callback. Only text can reconcile
    // an assistant text segment without imposing a rendering format.
    this.text.complete(
      typeof finalOutput === "string" ? finalOutput : undefined,
    );

    // A completed run should end open tools/agents as cancelled rather than
    // pretending Python sent successful end events it did not send.
    this.cancelOpenTools();
    this.cancelOpenAgents();
    this.abortApprovals();
  }

  private cancelOpenTools(): void {
    for (const call of this.openTools.values()) {
      this.emitTool({ phase: "cancelled", call });
    }
    this.openTools.clear();
  }

  private cancelOpenAgents(): void {
    for (const info of this.openAgents.values()) {
      this.emitAgent({ phase: "cancelled", info });
    }
    this.openAgents.clear();
  }

  private abortApprovals(): void {
    for (const approval of this.pendingApprovals.values()) {
      approval.controller.abort();
    }
    this.pendingApprovals.clear();
  }

  private emitTool(activity: ToolActivity): void {
    this.callHandler("onToolActivity", () =>
      this.callbacks.onToolActivity?.(activity),
    );
  }

  private emitAgent(activity: AgentActivity): void {
    this.callHandler("onAgentActivity", () =>
      this.callbacks.onAgentActivity?.(activity),
    );
  }

  private callHandler(handler: string, call: () => void): void {
    try {
      call();
    } catch (error) {
      // App callback failures are observability events. They must not corrupt
      // reducer state or prevent later terminal cleanup.
      this.callbacks.onDiagnostic({ kind: "handler-error", handler, error });
    }
  }
}

function normalizePlanStatus(status: string): PlanStepStatus {
  if (status === "completed") return "completed";
  if (status === "blocked") return "blocked";
  if (status === "in_progress" || status === "active") return "active";
  return "pending";
}

function normalizeApprovalDecision(decision: ApprovalDecision): {
  allowed: boolean;
  reason?: string;
} {
  if (typeof decision === "boolean") return { allowed: decision };

  return decision;
}

function denyAllApproval(): ApprovalDecision {
  return {
    allowed: false,
    reason: "No approval policy configured.",
  };
}

function parseToolArguments(argumentsText: string): unknown {
  try {
    return JSON.parse(argumentsText);
  } catch {
    return argumentsText;
  }
}
