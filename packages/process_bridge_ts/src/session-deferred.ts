/**
 * Promise plus externally held settlement functions.
 *
 * Bridge operations are initiated by a command write and settled later by an
 * event, process exit, or send failure. A deferred keeps that event-driven
 * settlement explicit without exposing mutable controller state to callers.
 */
export type Deferred<T> = {
  /** Promise returned to app code or internal waiters. */
  promise: Promise<T>;

  /** Resolve the promise when the matching backend event arrives. */
  resolve: (value: T) => void;

  /** Reject the promise on command failure or terminal session teardown. */
  reject: (error: Error) => void;
};

/** Build a deferred for a protocol operation settled by a later backend event. */
export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}
