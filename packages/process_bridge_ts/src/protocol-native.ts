import { z } from "zod";

const object: z.ZodRecord<z.ZodString, z.ZodUnknown> = z.record(
  z.string(),
  z.unknown(),
);
const nullableText: z.ZodNullable<z.ZodString> = z.string().nullable();
const lineage = {
  task_name: z.string(),
  task_id: z.string(),
  parent_task_id: nullableText,
  is_root: z.boolean(),
} as const;
const toolCall: z.ZodType<
  {
    id: string;
    type: "function";
    function: { name: string; arguments: string; [key: string]: unknown };
  } & Record<string, unknown>
> = z
  .object({
    id: z.string(),
    type: z.literal("function"),
    function: z
      .object({ name: z.string(), arguments: z.string() })
      .passthrough(),
  })
  .passthrough();
const usage: z.ZodType<
  {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } & Record<string, unknown>
> = z
  .object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
  })
  .passthrough();
const response: z.ZodType<
  {
    id: string;
    choices: Record<string, unknown>[];
    created: number;
    model: string;
    object: "chat.completion";
    usage?: z.infer<typeof usage> | null;
  } & Record<string, unknown>
> = z
  .object({
    id: z.string(),
    choices: z.array(object),
    created: z.number(),
    model: z.string(),
    object: z.literal("chat.completion"),
    usage: usage.nullable().optional(),
  })
  .passthrough();
const result: z.ZodType<
  {
    final_output: unknown;
    responses: z.infer<typeof response>[];
    turn_count: number;
    run_state: Record<string, unknown> | null;
  } & Record<string, unknown>
> = z
  .object({
    final_output: z.unknown(),
    responses: z.array(response),
    turn_count: z.number(),
    run_state: object.nullable(),
  })
  .passthrough();
const toolError: z.ZodType<
  { message: string; kind: string | null } & Record<string, unknown>
> = z.object({ message: z.string(), kind: nullableText }).passthrough();
const approvalRequest: z.ZodType<
  {
    tool_name: string;
    operation: string;
    cwd: string;
    path: string | null;
    command: string | null;
    skill_name: string | null;
    reason: string | null;
    run_id: string | null;
    agent_name: string | null;
    tool_call_id: string | null;
    metadata: Record<string, unknown>;
  } & Record<string, unknown>
> = z
  .object({
    tool_name: z.string(),
    operation: z.string(),
    cwd: z.string(),
    path: nullableText,
    command: nullableText,
    skill_name: nullableText,
    reason: nullableText,
    run_id: nullableText,
    agent_name: nullableText,
    tool_call_id: nullableText,
    metadata: object,
  })
  .passthrough();
const decision: z.ZodType<
  {
    outcome: "allow" | "deny" | "require_approval";
    reason: string | null;
  } & Record<string, unknown>
> = z
  .object({
    outcome: z.enum(["allow", "deny", "require_approval"]),
    reason: nullableText,
  })
  .passthrough();
const approvalRecord: z.ZodType<
  {
    request_id: string;
    request: z.infer<typeof approvalRequest>;
    approval_required_decision: z.infer<typeof decision>;
    status: "pending" | "resolved";
    final_decision: z.infer<typeof decision> | null;
  } & Record<string, unknown>
> = z
  .object({
    request_id: z.string(),
    request: approvalRequest,
    approval_required_decision: decision,
    status: z.enum(["pending", "resolved"]),
    final_decision: decision.nullable(),
  })
  .passthrough()
  .superRefine((record, context) => {
    if (
      record.status === "resolved" &&
      (record.final_decision === null ||
        record.final_decision.outcome === "require_approval")
    ) {
      context.addIssue({
        code: "custom",
        path: ["final_decision"],
        message: "Resolved approval requires an allow or deny decision.",
      });
    }
  });

/** Complete normalized model fields, with provider data retained recursively. */
export const MODEL_EVENT_SCHEMA: z.ZodType<
  {
    kind: string;
    raw: unknown;
    provider_event_type: string | null;
    item_index: number | null;
    item_type: string | null;
    text: string | null;
    tool_call_id: string | null;
    tool_call_index: number | null;
    arguments_delta: string | null;
    reasoning: unknown;
    reasoning_signature: string | null;
    response: z.infer<typeof response> | null;
    error: Record<string, unknown> | null;
  } & Record<string, unknown>
> = z
  .object({
    kind: z.string(),
    raw: z.unknown(),
    provider_event_type: nullableText,
    item_index: z.number().nullable(),
    item_type: nullableText,
    text: nullableText,
    tool_call_id: nullableText,
    tool_call_index: z.number().nullable(),
    arguments_delta: nullableText,
    reasoning: z.unknown(),
    reasoning_signature: nullableText,
    response: response.nullable(),
    error: object.nullable(),
  })
  .passthrough();

const modelKinds: ReadonlySet<string> = new Set([
  "provider",
  "text_delta",
  "tool_call_arguments_delta",
  "reasoning",
  "completed",
  "error",
]);
const modelEvent: z.ZodType<{ kind: string } & Record<string, unknown>> = z
  .object({ kind: z.string() })
  .passthrough()
  .superRefine((event, context) => {
    if (!modelKinds.has(event.kind)) return;
    const parsed = MODEL_EVENT_SCHEMA.safeParse(event);
    if (!parsed.success)
      parsed.error.issues.forEach((issue) => {
        context.addIssue({ ...issue });
      });
  });

/** Native source payload schemas. Each object keeps its extra fields. */
export const NATIVE_EVENT_SCHEMAS = {
  model_stream: z
    .object({ kind: z.literal("model_stream"), event: modelEvent })
    .passthrough(),
  agent_start: z
    .object({ kind: z.literal("agent_start"), ...lineage })
    .passthrough(),
  agent_end: z
    .object({
      kind: z.literal("agent_end"),
      ...lineage,
      result: result.nullable(),
    })
    .passthrough(),
  llm_start: z
    .object({
      kind: z.literal("llm_start"),
      ...lineage,
      messages: z.array(object),
    })
    .passthrough(),
  llm_end: z
    .object({ kind: z.literal("llm_end"), ...lineage, response })
    .passthrough(),
  tool_start: z
    .object({
      kind: z.literal("tool_start"),
      ...lineage,
      tool_call: toolCall,
      is_delegation: z.boolean(),
    })
    .passthrough(),
  tool_end: z
    .object({
      kind: z.literal("tool_end"),
      ...lineage,
      tool_call: toolCall,
      is_delegation: z.boolean(),
      result: z.unknown(),
      ok: z.boolean(),
      error: toolError.nullable(),
    })
    .passthrough(),
  tool_approval: z
    .object({
      kind: z.literal("tool_approval"),
      event: z.object({ record: approvalRecord }).passthrough(),
    })
    .passthrough(),
  handoff_start: z
    .object({
      kind: z.literal("handoff_start"),
      ...lineage,
      tool_call: toolCall,
      target_name: z.string(),
    })
    .passthrough(),
  handoff_end: z
    .object({
      kind: z.literal("handoff_end"),
      ...lineage,
      tool_call: toolCall,
      target_name: z.string(),
      result,
    })
    .passthrough(),
  state_snapshot: z
    .object({
      kind: z.literal("state_snapshot"),
      boundary: z.string(),
      snapshot: z
        .object({
          turn_count: z.number(),
          history_length: z.number(),
          response_count: z.number(),
          shim_state: object,
        })
        .passthrough(),
    })
    .passthrough(),
  plan_updated: z
    .object({
      kind: z.literal("plan_updated"),
      ...lineage,
      tool_call: toolCall,
      plan: z.array(
        z.object({ step: z.string(), status: z.string() }).passthrough(),
      ),
      explanation: nullableText,
    })
    .passthrough(),
} as const;

/** Known native event names with typed presentation support. */
export type NativeEventKind = keyof typeof NATIVE_EVENT_SCHEMAS;
/** Source payload for a known native event kind. */
export type NativePayload<K extends NativeEventKind> = z.infer<
  (typeof NATIVE_EVENT_SCHEMAS)[K]
>;
/** Full native record, including future event kinds and source extras. */
export type NativeRunEvent = {
  type: string;
  payload: Record<string, unknown>;
  [key: string]: unknown;
};
/** Normalized model event fields after boundary validation. */
export type NativeModelEvent = z.infer<typeof MODEL_EVENT_SCHEMA>;
/** Approval broker record used for execution and presentation. */
export type NativeApprovalRecord = z.infer<typeof approvalRecord>;
/** Complete source approval request for app permission UI. */
export type ApprovalRequestPayload = z.infer<typeof approvalRequest>;
/** Source token counts, including provider extensions. */
export type TokenUsage = z.infer<typeof usage>;
/** Source tool failure details. */
export type ToolErrorPayload = z.infer<typeof toolError>;

/** Validate known source fields without dropping unknown record or payload fields. */
export const NATIVE_RUN_EVENT_SCHEMA: z.ZodType<NativeRunEvent> = z
  .object({
    type: z.string(),
    payload: object,
  })
  .passthrough()
  .superRefine((event, context) => {
    if (!Object.hasOwn(NATIVE_EVENT_SCHEMAS, event.type)) {
      if (
        typeof event.payload.kind === "string" &&
        Object.hasOwn(NATIVE_EVENT_SCHEMAS, event.payload.kind)
      ) {
        context.addIssue({
          code: "custom",
          path: ["payload", "kind"],
          message: "Known native kind does not match record type.",
        });
      }
      return;
    }
    const parsed = NATIVE_EVENT_SCHEMAS[
      event.type as NativeEventKind
    ].safeParse(event.payload);
    if (!parsed.success)
      parsed.error.issues.forEach((issue) => {
        context.addIssue({ ...issue, path: ["payload", ...issue.path] });
      });
  });

/** Narrow an already decoded native record without rebuilding its payload. */
export function isNativeEvent<K extends NativeEventKind>(
  event: NativeRunEvent,
  kind: K,
): event is NativeRunEvent & { type: K; payload: NativePayload<K> } {
  return event.type === kind;
}

/** Narrow known model kinds after native boundary validation. */
export function isKnownModelEvent(event: {
  kind: string;
}): event is NativeModelEvent {
  return modelKinds.has(event.kind);
}
