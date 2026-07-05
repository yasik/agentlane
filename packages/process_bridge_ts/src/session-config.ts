import type { ConfigEvent } from "./protocol.ts";
import type { PendingCommandQueue } from "./session-pending-commands.ts";
import { ConfigureError } from "./session-types.ts";

/**
 * Controller services the config cache needs but should not own directly.
 *
 * Config decoding can close the whole session, and config announcements can
 * call app code. Keeping those effects injected makes this class about config
 * state and settlement only, not process lifecycle.
 */
type ConfigCallbacks<TConfig extends object> = {
  /**
   * Optional app decoder for the opaque backend document.
   *
   * This is where apps enforce their own config schema; the bridge only owns
   * transport-level settlement and cache timing.
   */
  decodeConfig?: (raw: Record<string, unknown>) => TConfig;

  /** Post-startup notification hook for backend-announced config changes. */
  onConfigChanged?: (config: Readonly<TConfig>) => void;

  /** Invoke app code behind the controller's handler-error diagnostics guard. */
  callAppHandler: (handler: string, call: () => void) => void;

  /** Close the session when TypeScript and Python disagree on config protocol. */
  failProtocol: (message: string, fields: readonly string[]) => void;
};

/**
 * Authoritative runtime config cache for one session.
 *
 * The bridge never predicts config locally. This helper only accepts backend
 * announcements, applies the app decoder, updates the cache, and optionally
 * notifies post-startup subscribers.
 */
export class SessionConfigState<
  TConfig extends object = Record<string, unknown>,
> {
  private readonly callbacks: ConfigCallbacks<TConfig>;
  private currentConfig: Readonly<TConfig> | undefined;

  /** Capture the controller callbacks used for decoding and app notification. */
  constructor(callbacks: ConfigCallbacks<TConfig>) {
    this.callbacks = callbacks;
  }

  /**
   * Latest backend-announced document.
   *
   * This is intentionally read-only and never predicted from a local patch. UI
   * code should render from this cache or from `onConfigChanged`, both of which
   * are sourced from Python acknowledgements.
   */
  get current(): Readonly<TConfig> | undefined {
    return this.currentConfig;
  }

  /**
   * Validate local patch hazards before JSON encoding.
   *
   * `JSON.stringify` drops `undefined` object values, which would turn
   * `configure({ model: undefined })` into `{}` and make the app think it sent
   * an explicit reset. Rejecting here keeps patch semantics visible.
   */
  validatePatch(patch: Record<string, unknown>): ConfigureError | null {
    const invalidKey = topLevelUndefinedKey(patch);
    if (invalidKey === undefined) return null;

    return new ConfigureError(
      "invalid",
      `Configure patch value must not be undefined: ${invalidKey}.`,
    );
  }

  /**
   * Accept one authoritative backend config document.
   *
   * Startup, reset, successful configure, and failed configure resyncs all use
   * this path so decoding and notification behavior cannot drift.
   */
  apply(
    raw: Record<string, unknown>,
    notify: boolean,
  ): Readonly<TConfig> | undefined {
    // The optional decoder is the app's lockstep contract with its Python
    // RuntimeConfigStore. A throw means frontend/backend drift, so failProtocol
    // closes the session instead of exposing a partially trusted document.
    const decoded = this.decode(raw);
    if (decoded === undefined) return undefined;

    this.currentConfig = decoded;
    if (notify) {
      this.callbacks.callAppHandler("onConfigChanged", () => {
        this.callbacks.onConfigChanged?.(decoded);
      });
    }

    return decoded;
  }

  /**
   * Settle one backend `config` event against the oldest pending configure.
   *
   * Ready/reset config announcements never arrive here; only the `config` event
   * that settles a `configure()` command does. A free-floating `config` event is
   * therefore protocol drift and must close the session loudly.
   */
  settleEvent(event: ConfigEvent, pendingCommands: PendingCommandQueue): void {
    if (!pendingCommands.has("configure")) {
      this.callbacks.failProtocol(
        "Unexpected config event without pending configure.",
        ["type"],
      );
      return;
    }

    let decoded: Readonly<TConfig> | undefined;
    if (event.config !== null) {
      // Failed settlements may still carry a truth snapshot. Apply it before
      // rejecting the promise so the app can immediately re-render backend
      // truth in the catch path.
      decoded = this.apply(event.config, true);
      if (decoded === undefined) return;
    }

    if (event.ok) {
      // The schema guarantees successful config events carry a document, but
      // keep the branch explicit so future edits cannot resolve with undefined.
      if (decoded === undefined) {
        this.callbacks.failProtocol(
          "Successful config event did not include config.",
          ["config"],
        );
        return;
      }

      const command = pendingCommands.take("configure");
      command?.resolveConfig?.(decoded);
      return;
    }

    // A failed settlement may omit a snapshot only when Python could not obtain
    // one. Keep the last known cache and reject the operation with the closed
    // transport-level error code.
    const error = event.error;
    if (error === null) {
      this.callbacks.failProtocol(
        "Failed config event did not include error.",
        ["error"],
      );
      return;
    }

    const command = pendingCommands.take("configure");
    command?.reject?.(new ConfigureError(error.code, error.message));
  }

  /** Decode app-owned config shape, converting decoder failures to protocol teardown. */
  private decode(raw: Record<string, unknown>): Readonly<TConfig> | undefined {
    try {
      return this.callbacks.decodeConfig?.(raw) ?? (raw as TConfig);
    } catch (error) {
      this.callbacks.failProtocol(
        `Invalid runtime config document: ${
          error instanceof Error ? error.message : String(error)
        }`,
        ["config"],
      );
      return undefined;
    }
  }
}

/**
 * Find the first top-level key that would silently disappear during JSON send.
 *
 * Nested values remain app-owned: config documents are opaque to the bridge, so
 * the local hazard we can safely reject is the top-level patch key vanishing.
 */
function topLevelUndefinedKey(
  patch: Record<string, unknown>,
): string | undefined {
  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined) return key;
  }

  return undefined;
}
