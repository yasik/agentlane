import type { BridgeChildLike } from "./channel.ts";

/**
 * Last-chance cleanup hook for a spawned backend process.
 *
 * Normal session shutdown goes through the bridge protocol. This hook only covers
 * host-process exit, where the app has no more async turns left to call
 * `session.close()` and the safest remaining action is killing the child.
 */
export class SessionExitHook {
  private cleanup: (() => void) | null = null;

  /** Install a process-exit listener that kills the child if the host exits. */
  install(child: BridgeChildLike): void {
    this.cleanup = (): void => {
      child.kill("SIGKILL");
    };
    process.on("exit", this.cleanup);
  }

  /** Remove the listener once startup fails or the session reaches a close path. */
  remove(): void {
    if (this.cleanup === null) return;

    process.removeListener("exit", this.cleanup);
    this.cleanup = null;
  }
}
