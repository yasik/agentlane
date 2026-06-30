# Process Bridge Stdio

This example runs a scripted Python AgentLane-compatible backend and a
TypeScript client over the process bridge. It does not require `OPENAI_API_KEY`.

Run the client from the repository root:

```bash
cd packages/process_bridge_ts
bun install
cd ../..
bun run examples/harness/process_bridge_stdio/client.ts
```

Expected output includes `ready`, `run_start`, `assistant_delta`,
`run_complete`, and `shutdown`.
