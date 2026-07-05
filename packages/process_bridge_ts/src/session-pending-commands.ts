import type { BridgeCommand } from "./protocol.ts";

/**
 * Command waiting for a backend acknowledgement or command-scoped error.
 *
 * The backend handles commands serially, so a command-scoped error belongs to
 * the oldest unsettled command. Configure commands also carry a resolver
 * because their acknowledgement event returns a typed config document.
 */
export type PendingCommand = {
  /** Wire command type; also the lookup key for typed acknowledgement events. */
  kind: BridgeCommand["type"];

  /** Reject the public operation, when this command has one. */
  reject?: (error: Error) => void;

  /** Resolve configure() with the decoded full document from the config event. */
  resolveConfig?: (config: Readonly<Record<string, unknown>>) => void;
};

/** Small FIFO helper for command/event correlation inside a live session. */
export class PendingCommandQueue {
  private readonly commands: PendingCommand[] = [];

  /** Track a command after it has been accepted for transport write. */
  push(command: PendingCommand): void {
    this.commands.push(command);
  }

  /**
   * Remove the oldest unsettled command.
   *
   * Command-scoped `error` events do not identify the rejected command. Python
   * handles commands serially, so the oldest pending command owns that error.
   */
  shift(): PendingCommand | undefined {
    return this.commands.shift();
  }

  /**
   * Remove the oldest pending command of a specific type.
   *
   * Acknowledgement events are typed (`run_start`, `reset`, `config`, ...), so
   * they should settle the matching command even if unrelated commands are also
   * pending. This preserves FIFO within each command kind without forcing all
   * commands to settle on a single event vocabulary.
   */
  take(kind: BridgeCommand["type"]): PendingCommand | undefined {
    const index = this.commands.findIndex(
      (command: PendingCommand): boolean => command.kind === kind,
    );
    if (index < 0) return undefined;

    const [command] = this.commands.splice(index, 1);
    return command;
  }

  /** Return whether any command of this kind is waiting for acknowledgement. */
  has(kind: BridgeCommand["type"]): boolean {
    return this.commands.some(
      (command: PendingCommand): boolean => command.kind === kind,
    );
  }

  /**
   * Remove one optimistic pending entry by identity.
   *
   * Used when stdin write fails after the controller queued the command but
   * before Python could possibly acknowledge it.
   */
  remove(command: PendingCommand): void {
    const index = this.commands.indexOf(command);
    if (index >= 0) this.commands.splice(index, 1);
  }

  /** Reject every unsettled command during terminal session teardown. */
  rejectAll(error: Error): void {
    for (const command of this.commands.splice(0)) {
      command.reject?.(error);
    }
  }
}
