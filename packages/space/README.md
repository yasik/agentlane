# AgentLane SPACE extension

`agentlane_space` runs AgentLane environment tools in an app-owned SPACE
sandbox. Bash uses the SPACE command API. File tools use AgentLane handlers
in a guest Python worker. Plan state and delegation stay in the host harness.
The root `agentlane` wheel includes this extension namespace.

The app owns sandbox creation, credentials, recovery, and deletion. The
sandbox and guest user permissions form the security boundary.

For installation, guest requirements, usage, and limits, see
[SPACE](../../docs/harness/sandboxes/space.md).

## Prepare a template

Use [template.toml.example](./template.toml.example) as a starting point for
your app's SPACE template. First prepare an Open Container Initiative (OCI)
image with Python 3.12, `/bin/bash`, `rg`, `tee`, and the same AgentLane wheel
as the host. Replace the `image` value with your image reference and its
digest. Your app owns template creation and sandbox deployment.

The template schema was checked against the SPACE source. The example has
not been tested against a live deployment.
