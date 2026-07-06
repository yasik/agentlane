# Runtime Configuration

Runtime configuration is for state owned by the Python agent instance, starting
with model settings. The app sends desired-state patches through
`session.configure(patch)`. The Python backend validates and applies them
through an app-owned `RuntimeConfigStore`, then announces the full authoritative
document back to TypeScript.

This is bridge-specific. Pure Python harness apps should configure their
`AgentDescriptor`, runner, tools, and model clients directly; they do not need a
process-bridge config store unless a TypeScript shell needs to change backend
state at runtime.

## Metadata vs Runtime Config

`ready.metadata` and `session.config` solve different problems:

1. `ready.metadata` is static startup context from the backend factory. It is
   emitted once in the `ready` event and exposed as `session.ready.metadata`.
   Use it for display and bootstrap facts such as app version, provider label,
   workspace path, approval mode, catalog version, or feature flags.
2. `session.config` is mutable backend state from `RuntimeConfigStore`. It is
   emitted as `ready.config` on startup, as the `config` settlement for each
   accepted `configure()` command, and as `reset.config` after reset. Use it for
   selected runtime values such as the current model id and selected model
   attributes.
3. Metadata does not settle commands, does not update after startup, and does not
   fire `onConfigChanged`. Config does all three. If the TypeScript app expects a
   value to change through `session.configure(...)`, it belongs in the config
   document, not only in metadata.
4. Both payloads are app-defined JSON objects. The bridge validates the envelope
   and size, but the app owns their inner schemas. Use `decodeConfig` for config
   documents; validate metadata in app code if you depend on specific keys.

For example, model display data can live in metadata, while the selected model
lives in config:

```python
def ready_metadata() -> dict[str, object]:
    return {
        "app_version": "1.2.3",
        "app_name": "my app",
    }
```

```ts
const session = await createAgentSession<ModelConfig>({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  decodeConfig: (raw) => configSchema.parse(raw),
});

appHeader.setProvider(String(session.ready.metadata.provider ?? "unknown"));

if (session.config) {
  modelPicker.setSelected(session.config.model);
}
```

## Minimal Config Store

```python
from pydantic import BaseModel, ConfigDict, ValidationError

from agentlane_process_bridge import (
    AgentBackend,
    ConfigRejectedError,
    RuntimeConfigStore,
)


class ModelConfigPatch(BaseModel):
    """Desired-state patch accepted from the TypeScript model picker."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None


class ModelSettings(RuntimeConfigStore):
    def __init__(self) -> None:
        self._document: dict[str, object] = {"model": "openai/gpt-5.5"}

    def snapshot(self) -> dict[str, object]:
        return dict(self._document)

    def apply(self, patch: dict[str, object]) -> dict[str, object]:
        try:
            parsed = ModelConfigPatch.model_validate(patch)
        except ValidationError as exc:
            raise ConfigRejectedError(str(exc)) from exc

        if parsed.model is not None:
            self._document = {**self._document, "model": parsed.model}

        return self.snapshot()


def create_backend() -> AgentBackend:
    return AgentBackend(agent=build_agent(), config=ModelSettings())
```

TypeScript app:

```ts
type ModelConfig = { model: string };
type ModelConfigPatch = { model?: string };

const session = await createAgentSession<ModelConfig, ModelConfigPatch>({
  backend: { app: "my_app.backend:create_backend", projectDir: "." },
  decodeConfig: (raw) => {
    if (typeof raw.model !== "string") throw new Error("model must be string");
    return { model: raw.model };
  },
  onConfigChanged: (config) => modelPicker.setSelected(config.model),
});

if (session.config) modelPicker.setSelected(session.config.model);
await session.configure({ model: "anthropic/claude-opus-4-8" });
```

`ready` carries the initial document when a store exists. The initial document
does not call `onConfigChanged`; read `session.config` after startup. Each
settled `configure()` emits exactly one `config` event. `reset` re-announces the
current document inside the reset event so the UI stays synchronized. Top-level
patch values must not be `undefined`; reset or disabled semantics should be
explicit app-defined values.

Config documents are opaque to the bridge but contract-critical: they are not
truncated. Documents that are not JSON-serializable or exceed the contract cap
fail loudly instead of being delivered as partial truth.

## Model Settings Propagation

Model settings move through one bridge-owned transport path and one app-owned
application path:

1. TypeScript calls `await session.configure({ model })`.
2. The TS session sends a `configure` command with an opaque JSON patch.
3. Python parses the command and calls `RuntimeConfigStore.apply(patch)`.
4. The app store validates the patch against the app catalog and applies the
   resulting selection to `AgentDescriptor.model` and
   `AgentDescriptor.model_args`.
5. Python emits one `config` event with the full applied document.
6. TypeScript decodes that document, updates `session.config`, fires
   `onConfigChanged`, and resolves `configure()`.
7. The next AgentLane model request reads the updated descriptor and uses the
   new model client / model args.

The bridge never learns what `model` means. It only guarantees command/event
ordering and full-document acknowledgement. The app owns the model catalog, the
provider client factory, and the mapping from selected attributes to model-call
kwargs.

For a model picker, keep one app-owned catalog file that both sides read:

```yaml
# my_app/models.yaml
models:
  - id: openai/gpt-5.5
    name: GPT-5.5
    default: true
    attributes:
      effort:
        default: medium
        options:
          low: { reasoning_effort: low }
          medium: { reasoning_effort: medium }
          high: { reasoning_effort: high }
  - id: anthropic/claude-opus-4-8
    name: Claude Opus 4.8
    attributes:
      thinking:
        default: disabled
        options:
          disabled: {}
          enabled: { thinking: { type: enabled, budget_tokens: 8192 } }
```

The TypeScript app renders names and attribute options from the catalog. The
Python store resolves the selected entry onto the `DefaultAgent` descriptor the
agent already reads on each model call:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config
from agentlane_litellm import Factory
from agentlane_process_bridge import (
    AgentBackend,
    ConfigRejectedError,
    RuntimeConfigStore,
)
from pydantic import BaseModel, ConfigDict, ValidationError


class ModelConfigPatch(BaseModel):
    """Desired-state model-selection patch from the TypeScript app."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    attributes: dict[str, str] | None = None


class ModelSettings(RuntimeConfigStore):
    def __init__(
        self,
        catalog: ModelCatalog,
        factory: Factory,
        descriptor: AgentDescriptor,
    ) -> None:
        self._catalog = catalog
        self._factory = factory
        self._descriptor = descriptor
        self._selection = catalog.default_selection()

    def snapshot(self) -> dict[str, object]:
        return self._selection.as_document()

    def apply(self, patch: dict[str, object]) -> dict[str, object]:
        try:
            parsed = ModelConfigPatch.model_validate(patch)
        except ValidationError as exc:
            raise ConfigRejectedError(str(exc)) from exc

        self._selection = self._catalog.resolve(self._selection, parsed)
        self._descriptor.model = self._factory.get_model_client(
            model=self._selection.model_id,
        )
        self._descriptor.model_args = self._selection.model_kwargs()
        return self.snapshot()


def create_backend() -> AgentBackend:
    catalog = ModelCatalog.load("my_app/models.yaml")
    factory = Factory(Config(api_key="", model=catalog.default_model_id))
    descriptor = AgentDescriptor(
        name="Assistant",
        instructions="You are a helpful assistant.",
    )
    settings = ModelSettings(catalog, factory, descriptor)
    settings.apply({})
    return AgentBackend(agent=DefaultAgent(descriptor=descriptor), config=settings)
```

`ModelCatalog` is app code: parse the YAML, validate ids/attributes/options,
and raise `ConfigRejectedError` for unknown selections. The bridge does not load
or transmit the catalog; it only carries the selected config document, for
example `{"model": "openai/gpt-5.5", "attributes": {"effort": "medium"}}`.
