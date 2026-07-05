import type { PendingCommand } from "./session-pending-commands.ts";
import type { SessionDiagnostic } from "./session-types.ts";
import { ConfigureError, RunError } from "./session-types.ts";

type CommandErrorHandlers = {
  /** Emit diagnostics for command errors that do not reject a public promise. */
  diagnostic: (diagnostic: SessionDiagnostic) => void;

  /** Reject the currently active run when the prompt command itself failed. */
  rejectRun: (error: Error) => void;
};

/**
 * Route a command-scoped backend error to the operation that owns it.
 *
 * Python emits command errors without a command id. Because its command loop is
 * serial, the oldest pending command is the owner. Public promises are rejected
 * only for operations that have a caller waiting; fire-and-forget commands fall
 * through to diagnostics.
 */
export function handlePendingCommandError(
  command: PendingCommand | undefined,
  message: string,
  handlers: CommandErrorHandlers,
): void {
  if (command === undefined) {
    handlers.diagnostic({ kind: "command-rejected", message });
    return;
  }

  if (command.kind === "prompt") {
    // Prompt rejection is the only command error that directly belongs to a
    // public run promise. Local prechecks catch the common misuse cases first.
    handlers.rejectRun(new RunError(message));
    return;
  }

  if (command.kind === "cancel" || command.kind === "reset") {
    command.reject?.(new RunError(message));
    return;
  }

  if (command.kind === "configure") {
    // This is the version-skew path: older Python backends answer
    // "Unknown command: configure" as a command error instead of emitting a
    // config settlement. Rejecting prevents configure() from hanging forever.
    command.reject?.(new ConfigureError("unsupported", message));
    return;
  }

  handlers.diagnostic({ kind: "command-rejected", message });
}
