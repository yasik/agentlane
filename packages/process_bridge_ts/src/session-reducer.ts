import type { BridgeEvent } from "./protocol.ts";
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
  request: BridgeEvent & { type: "approval_request" };
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
  private nextSyntheticToolId = 1;

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
      case "run_start":
        this.callbacks.onRunStarted();
        return;
      case "assistant_delta":
        this.pushText("assistant", event.text);
        return;
      case "reasoning_delta":
        this.pushText("reasoning", event.text);
        return;
      case "tool_start":
        this.toolStarted(event);
        return;
      case "tool_end":
        this.toolEnded(event);
        return;
      case "agent_start":
        this.agentStarted(event);
        return;
      case "agent_end":
        this.agentEnded(event);
        return;
      case "plan_updated":
        this.planUpdated(event);
        return;
      case "approval_request":
        this.approvalRequested(event);
        return;
      case "approval_resolved":
        this.approvalResolved(event);
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

  private toolStarted(event: BridgeEvent & { type: "tool_start" }): void {
    // Tool rows interrupt streamed prose in the UI; close the current text
    // segment before emitting the tool start.
    this.interruptText();
    const call = this.toolCallInfo(event);
    this.openTools.set(call.callId, call);
    this.emitTool({ phase: "start", call });
  }

  private toolEnded(event: BridgeEvent & { type: "tool_end" }): void {
    this.interruptText();
    const call = this.openToolCallForEnd(event) ?? this.toolCallInfo(event);
    this.openTools.delete(call.callId);
    this.emitTool({
      phase: "end",
      call,
      ok: event.ok,
      result: event.result,
      error: event.error,
    });
  }

  private openToolCallForEnd(
    event: BridgeEvent & { type: "tool_end" },
  ): ToolCallInfo | undefined {
    const exact = this.openTools.get(event.tool_call_id);
    if (exact !== undefined) return exact;

    // Older or synthetic tool-start events may not have a stable call id. Match
    // on the task-local identity before synthesizing a fresh end-only call.
    return [...this.openTools.values()].find(
      (call: ToolCallInfo): boolean =>
        call.tool === event.tool &&
        call.agent === event.agent &&
        call.taskId === event.task_id,
    );
  }

  private toolCallInfo(
    event: BridgeEvent & { type: "tool_start" | "tool_end" },
  ): ToolCallInfo {
    // Empty tool_call_id cannot be a reliable map key. Keep the wire id when it
    // exists so approval payloads can still be correlated exactly.
    const callId =
      event.tool_call_id.trim() === ""
        ? `synthetic-tool-${this.nextSyntheticToolId++}`
        : event.tool_call_id;
    return {
      callId,
      tool: event.tool,
      agent: event.agent,
      taskId: event.task_id,
      arguments: event.type === "tool_start" ? event.arguments : null,
      isPlan: event.is_plan,
      isDelegation: event.is_delegation,
    };
  }

  private agentStarted(event: BridgeEvent & { type: "agent_start" }): void {
    // Agent lifecycle is a structural boundary, but it does not invalidate the
    // active text stream; flush pending text without marking it done.
    this.flushText();
    const info = this.agentInfo(event);
    this.openAgents.set(info.taskId, info);
    this.emitAgent({ phase: "start", info });
  }

  private agentEnded(event: BridgeEvent & { type: "agent_end" }): void {
    this.flushText();
    const info = this.openAgents.get(event.task_id) ?? this.agentInfo(event);
    this.openAgents.delete(info.taskId);
    this.emitAgent({
      phase: "end",
      info,
      finalPreview: event.final_preview,
    });
  }

  private agentInfo(
    event: BridgeEvent & { type: "agent_start" | "agent_end" },
  ): AgentInfo {
    return {
      agent: event.agent,
      taskId: event.task_id,
      parentTaskId: event.parent_task_id,
      isRoot: event.is_root,
    };
  }

  private planUpdated(event: BridgeEvent & { type: "plan_updated" }): void {
    this.flushText();
    this.callHandler("onPlan", () =>
      this.callbacks.onPlan?.({
        agent: event.agent,
        taskId: event.task_id,
        explanation: event.explanation,
        steps: event.steps.map((step) => ({
          text: step.step,
          status: normalizePlanStatus(step.status),
          rawStatus: step.status,
        })),
      }),
    );
  }

  private approvalRequested(
    event: BridgeEvent & { type: "approval_request" },
  ): void {
    this.flushText();
    const controller = new AbortController();
    this.pendingApprovals.set(event.id, {
      controller,
      request: event,
      decidedByApp: false,
    });
    const policy = this.callbacks.approvals ?? denyAllApproval;

    // Approval policy code is app-owned and may be async. Keep it outside the
    // event dispatch stack so a slow modal cannot block protocol processing.
    Promise.resolve(
      policy({
        request: event.request,
        reason: event.reason,
        signal: controller.signal,
      }),
    )
      .then((decision: ApprovalDecision): void => {
        this.sendApprovalDecision(event.id, decision);
      })
      .catch((error: unknown): void => {
        this.callbacks.onDiagnostic({
          kind: "handler-error",
          handler: "approvals",
          error,
        });
        this.sendApprovalDecision(event.id, {
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
      // `approval_resolved` remains the source of truth. This flag only tells
      // the app whether its policy produced the resolution Python confirmed.
      pending.decidedByApp = true;
    }
  }

  private approvalResolved(
    event: BridgeEvent & { type: "approval_resolved" },
  ): void {
    this.flushText();
    const pending = this.pendingApprovals.get(event.id);
    this.pendingApprovals.delete(event.id);
    // Abort even after a normal app decision so any UI waiting on the signal can
    // dismiss once Python confirms the approval is no longer pending.
    pending?.controller.abort();
    this.callHandler("onApprovalResolved", () =>
      this.callbacks.onApprovalResolved?.({
        request: event.request,
        allowed: event.allowed,
        reason: event.reason,
        decidedByApp: pending?.decidedByApp ?? false,
      }),
    );
  }

  private completeRun(finalOutput: string): void {
    // A completed run should end open tools/agents as cancelled rather than
    // pretending Python sent successful end events it did not send.
    this.text.complete(finalOutput);
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
