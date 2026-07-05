import type { TextChunk } from "./session-types.ts";

/** How streamed text should be delivered to app callbacks. */
export type TextDelivery = "coalesced" | "immediate";

/** Stream kinds that can occupy one visible text segment. */
export type TextStreamKind = "assistant" | "reasoning";

/** Callback pair used by `TextStreamTracker` after segment assembly. */
export type TextStreamHandlers = {
  onAssistantText?: (chunk: TextChunk) => void;
  onReasoningText?: (chunk: TextChunk) => void;
};

type OpenSegment = {
  kind: TextStreamKind;
  segment: number;
  text: string;
  pendingDelta: string;
  timer: ReturnType<typeof setTimeout> | null;
};

/**
 * Tracks text segments and guarantees one terminal chunk per opened segment.
 *
 * The bridge emits model deltas, but apps need stable UI units. This tracker
 * converts contiguous assistant/reasoning deltas into numbered segments,
 * coalesces small deltas by default, and reconciles the final assistant segment
 * against `run_complete.final_output`.
 */
export class TextStreamTracker {
  private readonly delivery: TextDelivery;
  private readonly handlers: TextStreamHandlers;
  private nextSegment = 1;
  private active: OpenSegment | null = null;

  constructor(
    handlers: TextStreamHandlers,
    delivery: TextDelivery = "coalesced",
  ) {
    this.handlers = handlers;
    this.delivery = delivery;
  }

  push(kind: TextStreamKind, delta: string): void {
    if (this.active !== null && this.active.kind !== kind) {
      // Reasoning and assistant text are rendered as separate semantic rows.
      // Switching kind closes the previous row before opening the next.
      this.closeCurrent();
    }

    if (this.active === null) {
      this.active = {
        kind,
        segment: this.nextSegment,
        text: "",
        pendingDelta: "",
        timer: null,
      };
      this.nextSegment += 1;
    }

    this.active.text += delta;
    this.active.pendingDelta += delta;

    // Immediate mode preserves raw streaming cadence for tests and low-latency
    // UIs. Coalesced mode limits callback churn while preserving segment order.
    if (
      this.delivery === "immediate" ||
      shouldFlush(this.active.pendingDelta)
    ) {
      this.flush();
      return;
    }

    this.scheduleFlush();
  }

  interrupt(): void {
    this.closeCurrent();
  }

  flush(): void {
    const active = this.active;
    if (active === null || active.pendingDelta === "") return;

    this.clearTimer(active);
    const delta = active.pendingDelta;
    active.pendingDelta = "";
    this.emit(active, delta, false);
  }

  complete(finalAssistantText?: string): void {
    const active = this.active;

    if (active === null) {
      if (finalAssistantText) {
        // Some providers may only report the final output at completion. Emit a
        // one-chunk assistant segment so app code still sees the answer.
        const synthetic: OpenSegment = {
          kind: "assistant",
          segment: this.nextSegment,
          text: finalAssistantText,
          pendingDelta: "",
          timer: null,
        };
        this.nextSegment += 1;
        this.emit(synthetic, finalAssistantText, true);
      }
      return;
    }

    if (finalAssistantText !== undefined) {
      if (active.kind === "assistant") {
        // The completion event is authoritative. Reconcile any buffered or
        // provider-normalized text before marking the assistant segment done.
        this.closeCurrent(finalAssistantText);
        return;
      }

      // A reasoning segment was still open when the run completed. Close it
      // first, then synthesize or reconcile the final assistant answer.
      this.closeCurrent();
      this.complete(finalAssistantText);
      return;
    }

    this.closeCurrent();
  }

  dispose(): void {
    if (this.active !== null) {
      this.clearTimer(this.active);
    }
  }

  private closeCurrent(finalText?: string): void {
    const active = this.active;
    if (active === null) return;

    this.clearTimer(active);

    const text = finalText ?? active.text;
    const currentText = active.text;
    const pendingDelta = active.pendingDelta;
    active.pendingDelta = "";
    active.text = text;

    const delta = finalDelta(currentText, pendingDelta, text);
    this.emit(active, delta, true);
    this.active = null;
  }

  private scheduleFlush(): void {
    const active = this.active;
    if (active === null || active.timer !== null) return;

    // The timer is per active segment. Closing or flushing a segment clears it
    // so late callbacks cannot emit against stale text.
    active.timer = setTimeout(() => {
      active.timer = null;
      this.flush();
    }, 120);
  }

  private clearTimer(segment: OpenSegment): void {
    if (segment.timer === null) return;

    clearTimeout(segment.timer);
    segment.timer = null;
  }

  private emit(segment: OpenSegment, delta: string, done: boolean): void {
    const chunk: TextChunk = {
      delta,
      text: segment.text,
      segment: segment.segment,
      done,
    };

    if (segment.kind === "assistant") {
      this.handlers.onAssistantText?.(chunk);
      return;
    }

    this.handlers.onReasoningText?.(chunk);
  }
}

function shouldFlush(delta: string): boolean {
  // Flush early for large buffers and for sentence-ish boundaries once enough
  // text has accumulated to make coalescing visible.
  return delta.length >= 96 || (delta.length >= 32 && /[.!?]\s$/.test(delta));
}

function finalDelta(
  currentText: string,
  pendingDelta: string,
  finalText: string,
): string {
  if (finalText === currentText) return pendingDelta;

  if (finalText.startsWith(currentText)) {
    // Preserve any unflushed delta and append only the authoritative suffix.
    return pendingDelta + finalText.slice(currentText.length);
  }

  // Provider-normalized final output diverged from the streamed text. Emit the
  // final text as the terminal delta so the app's last chunk is authoritative.
  return finalText;
}
