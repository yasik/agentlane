"""Tests for Claude-Code-style markdown agent definitions (`agents.definitions`)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

import agentlane.harness.agents.definitions as definitions_pkg
from agentlane.harness import (
    INHERIT_TOOLS,
    OVERRIDE_TOOLS,
    AgentDescriptor,
    InheritTools,
    RestrictTools,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.agents.definitions import (
    AGENT_MAX_DESCRIPTION_LENGTH,
    AGENT_MAX_INSTRUCTIONS_LINES,
    AgentFileError,
    FactoryModelResolver,
    SubagentLink,
    descriptor_from_markdown,
    parse_agent_file,
    resolve_tool_config,
)
from agentlane.harness.shims import ExcludeToolsShim, PreparedTurn
from agentlane.models import (
    Config,
    Factory,
    Model,
    ModelResponse,
    ModelTracing,
    Tools,
)

from .tools_test_utils import (
    SequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
    run_state,
)


def _write_agent_md(path: Path, frontmatter: str, body: str = "System body.") -> Path:
    """Write one `AGENT.md` file with the given frontmatter block and body."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return path


class _MappingResolver:
    """Structural `ModelResolver` backed by a spec -> model mapping."""

    def __init__(self, models: dict[str, Model[ModelResponse]]) -> None:
        self._models = models
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def resolve(
        self,
        model_spec: str,
        *,
        model_args: dict[str, Any],
    ) -> Model[ModelResponse]:
        self.calls.append((model_spec, dict(model_args)))
        return self._models[model_spec]


class _RecordingFactory(Factory[ModelResponse]):
    """Factory stub that records build kwargs and returns a fixed client."""

    def __init__(self, client: Model[ModelResponse]) -> None:
        super().__init__(Config(api_key="key", model="default/model"))
        self._client = client
        self.calls: list[dict[str, Any]] = []

    def get_model_client(
        self,
        tracing: ModelTracing = ModelTracing.DISABLED,
        **kwargs: Any,
    ) -> Model[ModelResponse]:
        del tracing
        self.calls.append(dict(kwargs))
        return self._client


# --------------------------------------------------------------------------- #
# Phase 1 — parser
# --------------------------------------------------------------------------- #


def test_parse_agent_file_returns_manifest_and_body(tmp_path: Path) -> None:
    path = _write_agent_md(
        tmp_path / "a.md",
        "\n".join(
            [
                "name: code-reviewer",
                "description: Reviews diffs.",
                "model: anthropic/claude-sonnet-4-5",
                "model_args:",
                "  temperature: 0.2",
                "  max_tokens: 4096",
                "tools: [read, grep, bash]",
                "disallowedTools: write",
            ]
        ),
        body="You are a reviewer.",
    )

    parsed = parse_agent_file(path)

    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    manifest = parsed.manifest
    assert manifest.name == "code-reviewer"
    assert manifest.description == "Reviews diffs."
    assert manifest.model_spec == "anthropic/claude-sonnet-4-5"
    assert manifest.model_args == {"temperature": 0.2, "max_tokens": 4096}
    assert manifest.allowed_tools == ("read", "grep", "bash")
    assert manifest.disallowed_tools == ("write",)
    assert manifest.source_path == path.resolve()
    assert parsed.instructions == "You are a reviewer."


def test_parse_agent_file_name_absent_is_none(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "description: No name here.")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.name is None


def test_parse_agent_file_captures_model_spec_verbatim(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "model: openai/gpt-5.1")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.model_spec == "openai/gpt-5.1"


def test_parse_agent_file_inherit_model_becomes_none(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "model: inherit")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.model_spec is None


def test_parse_agent_file_preserves_conflicting_model_args(tmp_path: Path) -> None:
    path = _write_agent_md(
        tmp_path / "a.md",
        "\n".join(
            [
                "model: openai/gpt-5.1",
                "model_args:",
                "  temperature: 0.2",
                "  reasoning_effort: high",
            ]
        ),
    )

    parsed = parse_agent_file(path)

    if parsed is None:
        raise AssertionError("expected a parsed agent file")

    # Conflicting args pass through untouched; the mutual-exclusion rule is
    # enforced at the model call, not swallowed here.
    assert parsed.manifest.model_args == {
        "temperature": 0.2,
        "reasoning_effort": "high",
    }


def test_parse_agent_file_truncates_oversized_description(tmp_path: Path) -> None:
    long_description = "x" * (AGENT_MAX_DESCRIPTION_LENGTH + 100)
    path = _write_agent_md(tmp_path / "a.md", f"description: {long_description}")

    parsed = parse_agent_file(path)

    if parsed is None:
        raise AssertionError("expected a parsed agent file")

    assert parsed.manifest.description is not None
    assert len(parsed.manifest.description) == AGENT_MAX_DESCRIPTION_LENGTH


def test_parse_agent_file_tools_omitted_is_none(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "name: a")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.allowed_tools is None


def test_parse_agent_file_empty_tools_list_is_empty_tuple(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "tools: []")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.allowed_tools == ()


def test_parse_agent_file_tools_comma_string_parsed(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "tools: read, grep , bash")
    parsed = parse_agent_file(path)
    if parsed is None:
        raise AssertionError("expected a parsed agent file")
    assert parsed.manifest.allowed_tools == ("read", "grep", "bash")


def test_parse_agent_file_returns_none_for_missing_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("no frontmatter here\n", encoding="utf-8")
    assert parse_agent_file(path) is None


def test_parse_agent_file_returns_none_for_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("---\nname: [unterminated\n---\nbody\n", encoding="utf-8")
    assert parse_agent_file(path) is None


def test_parse_agent_file_returns_none_for_non_mapping_frontmatter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.md"
    path.write_text("---\n- just\n- a\n- list\n---\nbody\n", encoding="utf-8")
    assert parse_agent_file(path) is None


def test_parse_agent_file_truncates_oversized_instructions(tmp_path: Path) -> None:
    line_count = AGENT_MAX_INSTRUCTIONS_LINES + 50
    body = "\n".join(f"line {index}" for index in range(line_count))
    path = _write_agent_md(tmp_path / "a.md", "name: verbose", body=body)

    parsed = parse_agent_file(path)

    if parsed is None:
        raise AssertionError("expected a parsed agent file")

    # The frontmatter is still parsed; only the body is bounded, with a pointer.
    assert parsed.manifest.name == "verbose"
    assert len(parsed.instructions.splitlines()) < line_count
    assert "truncated" in parsed.instructions.lower()
    assert str(path.resolve()) in parsed.instructions


def test_parse_agent_file_propagates_read_error_for_missing_file(
    tmp_path: Path,
) -> None:
    # A missing file is a user-fixable system error: it propagates rather than
    # being swallowed into a silent None.
    with pytest.raises(FileNotFoundError):
        parse_agent_file(tmp_path / "missing.md")


# --------------------------------------------------------------------------- #
# Phase 2 — tool resolution
# --------------------------------------------------------------------------- #


def test_resolve_tool_config_omitted_inherits() -> None:
    config, shims = resolve_tool_config(None, ())
    assert config is INHERIT_TOOLS
    assert shims == ()


def test_resolve_tool_config_allowlist_restricts() -> None:
    config, shims = resolve_tool_config(("read", "grep"), ())
    assert isinstance(config, RestrictTools)
    assert config.names == frozenset({"read", "grep"})
    assert shims == ()


def test_resolve_tool_config_empty_overrides() -> None:
    config, shims = resolve_tool_config((), ())
    assert config is OVERRIDE_TOOLS
    assert shims == ()


def test_resolve_tool_config_denylist_adds_shim() -> None:
    config, shims = resolve_tool_config(None, ("write",))
    assert config is INHERIT_TOOLS
    assert len(shims) == 1
    shim = shims[0]
    assert isinstance(shim, ExcludeToolsShim)
    assert shim.excluded_names == frozenset({"write"})


def test_exclude_tools_shim_excludes_named_tools_each_turn() -> None:
    tools = Tools(tools=(echo_tool("read"), echo_tool("write")))
    shim = ExcludeToolsShim(names={"write"})
    turn = PreparedTurn(run_state=run_state(), tools=tools, model_args=None)

    asyncio.run(shim.prepare_turn(turn))

    if turn.tools is None:
        raise AssertionError("expected remaining tools")
    assert [tool.name for tool in turn.tools.normalized_tools] == ["read"]


# --------------------------------------------------------------------------- #
# Phase 3 — model seam
# --------------------------------------------------------------------------- #


def test_factory_model_resolver_builds_client_from_spec() -> None:
    client = SequenceModel([])
    factory = _RecordingFactory(client)
    resolver = FactoryModelResolver(factory=factory)

    resolved = resolver.resolve("anthropic/claude-x", model_args={})

    assert resolved is client
    assert factory.calls == [{"model": "anthropic/claude-x"}]


def test_factory_model_resolver_does_not_fold_model_args_into_client() -> None:
    factory = _RecordingFactory(SequenceModel([]))
    resolver = FactoryModelResolver(factory=factory)

    resolver.resolve("openai/gpt-5.1", model_args={"temperature": 0.2})

    assert factory.calls == [{"model": "openai/gpt-5.1"}]


def test_definitions_package_does_not_import_provider_packages() -> None:
    package_dir = Path(definitions_pkg.__file__).resolve().parent
    forbidden = ("agentlane_litellm", "agentlane_openai", "import litellm", "litellm.")
    for module_path in package_dir.glob("*.py"):
        text = module_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{module_path.name} must not reference {token}"


# --------------------------------------------------------------------------- #
# Phase 4 — loader / orchestration
# --------------------------------------------------------------------------- #


def test_descriptor_from_markdown_maps_fields(tmp_path: Path) -> None:
    path = _write_agent_md(
        tmp_path / "a.md",
        "\n".join(
            [
                "name: reviewer",
                "description: Reviews code.",
                "model_args:",
                "  temperature: 0.1",
                "tools: [read]",
                "disallowedTools: write",
            ]
        ),
        body="Be precise.",
    )

    descriptor = descriptor_from_markdown(path)

    assert descriptor.name == "reviewer"
    assert descriptor.description == "Reviews code."
    assert descriptor.instructions == "Be precise."
    assert descriptor.model is None
    assert descriptor.model_args == {"temperature": 0.1}
    assert isinstance(descriptor.tools, RestrictTools)
    assert descriptor.tools.names == frozenset({"read"})
    assert descriptor.shims is not None
    assert any(isinstance(shim, ExcludeToolsShim) for shim in descriptor.shims)


def test_descriptor_from_markdown_resolves_model_with_resolver(tmp_path: Path) -> None:
    client = SequenceModel([])
    resolver = _MappingResolver({"test/model": client})
    path = _write_agent_md(tmp_path / "a.md", "name: a\nmodel: test/model")

    descriptor = descriptor_from_markdown(path, model_resolver=resolver)

    assert descriptor.model is client
    assert resolver.calls == [("test/model", {})]


def test_descriptor_from_markdown_inherit_model_stays_none(tmp_path: Path) -> None:
    resolver = _MappingResolver({"test/model": SequenceModel([])})
    path = _write_agent_md(tmp_path / "a.md", "name: a\nmodel: inherit")

    descriptor = descriptor_from_markdown(path, model_resolver=resolver)

    assert descriptor.model is None
    assert resolver.calls == []


def test_descriptor_from_markdown_attaches_subagent_as_tool(tmp_path: Path) -> None:
    _write_agent_md(tmp_path / "helper.md", "name: helper\ndescription: A helper.")
    parent = _write_agent_md(tmp_path / "parent.md", "name: parent")

    descriptor = descriptor_from_markdown(
        parent,
        subagents=[tmp_path / "helper.md"],
    )

    assert descriptor.handoffs is None
    assert isinstance(descriptor.tools, InheritTools)
    child_tools = descriptor.tools.tools
    if child_tools is None:
        raise AssertionError("expected the sub-agent tool to be exposed")
    assert "helper" in [tool.name for tool in child_tools.normalized_tools]


def test_descriptor_from_markdown_attaches_subagent_as_handoff(tmp_path: Path) -> None:
    _write_agent_md(tmp_path / "helper.md", "name: helper")
    parent = _write_agent_md(tmp_path / "parent.md", "name: parent")

    descriptor = descriptor_from_markdown(
        parent,
        subagent_link=SubagentLink.HANDOFF,
        subagents=[tmp_path / "helper.md"],
    )

    if descriptor.handoffs is None:
        raise AssertionError("expected a handoff sub-agent")
    assert [child.name for child in descriptor.handoffs] == ["helper"]


def test_descriptor_from_markdown_child_model_args_are_independent(
    tmp_path: Path,
) -> None:
    _write_agent_md(
        tmp_path / "helper.md",
        "name: helper\nmodel_args:\n  temperature: 0.9",
    )
    parent = _write_agent_md(
        tmp_path / "parent.md",
        "name: parent\nmodel_args:\n  temperature: 0.1",
    )

    descriptor = descriptor_from_markdown(
        parent,
        subagent_link=SubagentLink.HANDOFF,
        subagents=[tmp_path / "helper.md"],
    )

    assert descriptor.model_args == {"temperature": 0.1}
    if descriptor.handoffs is None:
        raise AssertionError("expected a handoff sub-agent")
    assert descriptor.handoffs[0].model_args == {"temperature": 0.9}


def test_descriptor_from_markdown_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        descriptor_from_markdown(tmp_path / "missing.md")


def test_descriptor_from_markdown_raises_on_unparseable_file(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("no frontmatter\n", encoding="utf-8")
    with pytest.raises(AgentFileError):
        descriptor_from_markdown(path)


def test_descriptor_from_markdown_raises_on_cycle(tmp_path: Path) -> None:
    parent = _write_agent_md(tmp_path / "parent.md", "name: parent")
    with pytest.raises(AgentFileError, match="cycle"):
        descriptor_from_markdown(parent, subagents=[parent])


def test_descriptor_from_markdown_raises_on_duplicate_subagent_tool_names(
    tmp_path: Path,
) -> None:
    # Two distinct files whose names normalize to the same delegation tool name
    # would silently shadow each other and make runtime dispatch ambiguous.
    _write_agent_md(tmp_path / "a.md", "name: code-review")
    _write_agent_md(tmp_path / "b.md", "name: Code Review")
    parent = _write_agent_md(tmp_path / "parent.md", "name: parent")

    with pytest.raises(AgentFileError, match="duplicate sub-agent"):
        descriptor_from_markdown(
            parent,
            subagents=[tmp_path / "a.md", tmp_path / "b.md"],
        )


# --------------------------------------------------------------------------- #
# Programmatic sub-agents on DefaultAgent (subagents=, no manual as_tool)
# --------------------------------------------------------------------------- #


def test_default_agent_attaches_subagents_as_tools() -> None:
    child = AgentDescriptor(name="med-safety")
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="triage-lead"),
        subagents=[child],
    )

    tools = agent.resolved_descriptor.tools
    assert isinstance(tools, InheritTools)
    if tools.tools is None:
        raise AssertionError("expected the sub-agent tool to be attached")
    assert "med_safety" in [tool.name for tool in tools.tools.normalized_tools]


def test_default_agent_attaches_subagents_as_handoffs() -> None:
    child = AgentDescriptor(name="escalation")
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="triage-lead"),
        subagents=[child],
        subagent_link=SubagentLink.HANDOFF,
    )

    assert agent.resolved_descriptor.handoffs == (child,)


def test_default_agent_subagents_rejects_markdown_path() -> None:
    with pytest.raises(TypeError, match="from_markdown"):
        DefaultAgent(
            descriptor=AgentDescriptor(name="triage-lead"),
            subagents=["med_safety.md"],  # type: ignore[list-item]
        )


def test_default_agent_subagents_rejects_duplicate_tool_names() -> None:
    with pytest.raises(AgentFileError, match="duplicate sub-agent"):
        DefaultAgent(
            descriptor=AgentDescriptor(name="triage-lead"),
            subagents=[
                AgentDescriptor(name="med-safety"),
                AgentDescriptor(name="Med Safety"),
            ],
        )


# --------------------------------------------------------------------------- #
# Phase 6 — classmethods
# --------------------------------------------------------------------------- #


def test_descriptor_from_markdown_builds_descriptor(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "name: helper", body="Help out.")
    descriptor = descriptor_from_markdown(path)
    assert isinstance(descriptor, AgentDescriptor)
    assert descriptor.name == "helper"
    assert descriptor.instructions == "Help out."


def test_descriptor_from_markdown_generates_name_when_absent(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "description: anonymous")
    descriptor = descriptor_from_markdown(path)
    assert descriptor.name.strip() != ""
    assert "-" in descriptor.name


def test_default_agent_from_markdown_raises_without_model(tmp_path: Path) -> None:
    path = _write_agent_md(tmp_path / "a.md", "name: a")
    with pytest.raises(AgentFileError, match="model"):
        DefaultAgent.from_markdown(path)


def test_default_agent_from_markdown_model_param_satisfies_check(
    tmp_path: Path,
) -> None:
    client = SequenceModel([])
    path = _write_agent_md(tmp_path / "a.md", "name: a")
    agent = DefaultAgent.from_markdown(path, model=client)
    assert agent.resolved_descriptor.model is client


def test_default_agent_from_markdown_resolver_satisfies_check(tmp_path: Path) -> None:
    client = SequenceModel([])
    resolver = _MappingResolver({"test/model": client})
    path = _write_agent_md(tmp_path / "a.md", "name: a\nmodel: test/model")
    agent = DefaultAgent.from_markdown(path, model_resolver=resolver)
    assert agent.resolved_descriptor.model is client


def test_default_agent_from_markdown_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DefaultAgent.from_markdown(tmp_path / "missing.md")


def test_default_agent_from_markdown_runs_with_sequence_model(tmp_path: Path) -> None:
    client = SequenceModel([make_assistant_response("done")])
    path = _write_agent_md(
        tmp_path / "a.md",
        "name: greeter",
        body="You greet politely.",
    )
    agent = DefaultAgent.from_markdown(path, model=client)

    result = asyncio.run(agent.run("hi"))

    assert result.final_output == "done"
    first_call = client.calls[0]
    assert any("You greet politely." in str(message) for message in first_call)


def test_default_agent_from_markdown_subagent_runs_as_subroutine(
    tmp_path: Path,
) -> None:
    parent_model = SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(tool_id="t1", name="helper", arguments="{}")
                ],
            ),
            make_assistant_response("final answer"),
        ]
    )
    child_model = SequenceModel([make_assistant_response("child result")])
    resolver = _MappingResolver(
        {"test/parent": parent_model, "test/child": child_model}
    )

    _write_agent_md(
        tmp_path / "helper.md",
        "name: helper\ndescription: A helper.\nmodel: test/child",
        body="You are the helper.",
    )
    parent = _write_agent_md(
        tmp_path / "parent.md",
        "name: parent\nmodel: test/parent",
        body="You are the parent.",
    )

    agent = DefaultAgent.from_markdown(
        parent,
        model_resolver=resolver,
        subagents=[tmp_path / "helper.md"],
    )

    result = asyncio.run(agent.run("go"))

    assert result.final_output == "final answer"
    assert len(child_model.calls) == 1
