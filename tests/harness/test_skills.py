import asyncio
import json
import logging
from pathlib import Path

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from agentlane.harness import (
    AgentDescriptor,
    RunnerHooks,
    RunResult,
    RunState,
    ShimState,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.skills import (
    FilesystemSkillLoader,
    LoadedSkill,
    SkillCatalog,
    SkillLoader,
    SkillManifest,
    SkillResource,
    SkillsShim,
)
from agentlane.harness.skills._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
)
from agentlane.harness.skills._parser import parse_skill_file
from agentlane.harness.skills._prompt import (
    SkillsSystemPromptContext,
    render_loaded_skill,
    render_skills_system_prompt,
)
from agentlane.harness.skills._shim import ActivateSkillInput
from agentlane.models import MessageDict, Model, ModelResponse, ToolCall, Tools
from agentlane.runtime import CancellationToken


def _make_tool_call(
    *,
    tool_id: str,
    arguments: str,
    name: str,
) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


def _assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "skills-response",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[MessageDict]] = []
        self.call_options: list[dict[str, object]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del cancellation_token
        self.calls.append([dict(message) for message in messages])
        self.call_options.append(
            {
                "extra_call_args": extra_call_args,
                "schema": schema,
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        if not self._outcomes:
            raise AssertionError("Expected one queued model response.")
        return self._outcomes.pop(0)


class _MemorySkillLoader(SkillLoader):
    def __init__(self, loaded_skill: LoadedSkill) -> None:
        self._loaded_skill = loaded_skill
        self.discover_calls = 0
        self.load_calls: list[str] = []

    async def discover(self) -> tuple[SkillManifest, ...]:
        self.discover_calls += 1
        return (self._loaded_skill.manifest,)

    async def load(self, name: str) -> LoadedSkill:
        self.load_calls.append(name)
        if name != self._loaded_skill.manifest.name:
            raise KeyError(name)
        return self._loaded_skill


class _EmptySkillLoader(SkillLoader):
    async def discover(self) -> tuple[SkillManifest, ...]:
        return ()

    async def load(self, name: str) -> LoadedSkill:
        raise KeyError(name)


class _SkillActivationRecordingHooks(RunnerHooks):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del state
        self.events.append(("agent_start", getattr(task, "name", type(task).__name__)))

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        del task
        if tool_call.function.name != "activate_skill":
            return
        self.events.append(("tool_start", tool_call.function.arguments))

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        del task
        if tool_call.function.name != "activate_skill":
            return
        status = (
            "loaded"
            if isinstance(result, str) and "<skill_content" in result
            else "other"
        )
        self.events.append(("tool_end", status))

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        del task
        final_output = "" if result is None else str(result.final_output)
        self.events.append(("agent_end", final_output))


def _write_skill(
    *,
    root: Path,
    name: str,
    description: str,
    body: str,
    metadata: str | None = None,
    compatibility: str | None = None,
) -> Path:
    skill_root = root / name
    skill_root.mkdir(parents=True)

    frontmatter_lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    if compatibility is not None:
        frontmatter_lines.append(f"compatibility: {compatibility}")
    if metadata is not None:
        frontmatter_lines.extend(
            [
                "metadata:",
                f"  author: {metadata}",
            ]
        )
    frontmatter_lines.append("---")

    (skill_root / "SKILL.md").write_text(
        "\n".join(frontmatter_lines) + "\n\n" + body + "\n",
        encoding="utf-8",
    )
    return skill_root


def test_parse_skill_file_returns_manifest_and_body(tmp_path: Path) -> None:
    skill_root = _write_skill(
        root=tmp_path,
        name="refund-policy",
        description="Explain the refund policy when users ask about returns.",
        body="# Refund Policy\n\nUse this skill for return-window questions.",
        metadata="agentlane",
        compatibility="Requires local files only",
    )
    (skill_root / "references").mkdir()
    (skill_root / "references" / "policy.md").write_text(
        "Reference",
        encoding="utf-8",
    )

    parsed = parse_skill_file(skill_root / "SKILL.md")
    if parsed is None:
        raise AssertionError("Expected skill file to parse.")

    assert parsed.manifest.name == "refund-policy"
    assert parsed.manifest.description == (
        "Explain the refund policy when users ask about returns."
    )
    assert parsed.manifest.root == skill_root.resolve()
    assert parsed.manifest.skill_file == (skill_root / "SKILL.md").resolve()
    assert parsed.manifest.metadata == {"author": "agentlane"}
    assert parsed.manifest.compatibility == "Requires local files only"
    assert parsed.instructions == (
        "# Refund Policy\n\nUse this skill for return-window questions."
    )


def test_parse_skill_file_trims_oversize_fields_and_body(tmp_path: Path) -> None:
    description = "d" * (SKILL_MAX_DESCRIPTION_LENGTH + 25)
    compatibility = "c" * (SKILL_MAX_COMPATIBILITY_LENGTH + 25)
    body = "\n".join(f"line {index}" for index in range(SKILL_MAX_FILE_LINES - 10))

    skill_root = _write_skill(
        root=tmp_path,
        name="refund-policy",
        description=description,
        compatibility=compatibility,
        body=body,
    )

    parsed = parse_skill_file(skill_root / "SKILL.md")
    if parsed is None:
        raise AssertionError("Expected skill file to parse.")

    assert len(parsed.manifest.description) == SKILL_MAX_DESCRIPTION_LENGTH
    assert parsed.manifest.description == description[:SKILL_MAX_DESCRIPTION_LENGTH]
    if parsed.manifest.compatibility is None:
        raise AssertionError("Expected trimmed compatibility.")
    assert len(parsed.manifest.compatibility) == SKILL_MAX_COMPATIBILITY_LENGTH
    assert (
        parsed.manifest.compatibility == compatibility[:SKILL_MAX_COMPATIBILITY_LENGTH]
    )
    assert "line 0" in parsed.instructions
    assert f"line {SKILL_MAX_FILE_LINES - 11}" in parsed.instructions


def test_parse_skill_file_warns_but_keeps_non_compliant_skill_name(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    skill_root = _write_skill(
        root=tmp_path,
        name="refund--policy",
        description="Handle refund questions.",
        body="# Refund Policy",
    )

    with caplog.at_level(logging.WARNING):
        parsed = parse_skill_file(skill_root / "SKILL.md")

    if parsed is None:
        raise AssertionError("Expected malformed name to warn but still parse.")
    assert parsed.manifest.name == "refund--policy"
    assert "consecutive hyphens" in caplog.text


def test_parse_skill_file_returns_none_for_missing_frontmatter(tmp_path: Path) -> None:
    skill_root = tmp_path / "refund-policy"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text("# Refund Policy\n", encoding="utf-8")

    assert parse_skill_file(skill_root / "SKILL.md") is None


def test_parse_skill_file_returns_none_for_non_mapping_frontmatter(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "refund-policy"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n- item\n---\n\n# Refund Policy\n",
        encoding="utf-8",
    )

    assert parse_skill_file(skill_root / "SKILL.md") is None


def test_parse_skill_file_returns_none_for_empty_required_fields(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "refund-policy"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: \ndescription: \n---\n\n# Refund Policy\n",
        encoding="utf-8",
    )

    assert parse_skill_file(skill_root / "SKILL.md") is None


def test_parse_skill_file_returns_none_when_file_exceeds_size_limit(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "refund-policy"
    skill_root.mkdir(parents=True)
    body = "\n".join(f"line {index}" for index in range(SKILL_MAX_FILE_LINES + 50))
    (skill_root / "SKILL.md").write_text(
        (
            "---\n"
            "name: refund-policy\n"
            "description: Handle refund questions.\n"
            "---\n\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )

    assert parse_skill_file(skill_root / "SKILL.md") is None


def test_filesystem_skill_loader_loads_skill_by_manifest_name_when_directory_differs(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    skill_root = tmp_path / "refund-policy-dir"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        (
            "---\n"
            "name: refund-policy\n"
            "description: Handle refund questions.\n"
            "---\n\n"
            "# Refund Policy\n"
        ),
        encoding="utf-8",
    )

    loader = FilesystemSkillLoader(
        roots=(tmp_path,),
        include_default_roots=False,
    )

    async def scenario() -> None:
        with caplog.at_level(logging.WARNING):
            manifests = await loader.discover()
            loaded = await loader.load("refund-policy")

        assert [manifest.name for manifest in manifests] == ["refund-policy"]
        assert loaded.manifest.name == "refund-policy"
        assert loaded.manifest.root == skill_root.resolve()
        assert "does not match parent directory" in caplog.text

    asyncio.run(scenario())


def test_filesystem_skill_loader_discovers_valid_skills_and_lists_resources(
    tmp_path: Path,
) -> None:
    valid_root = _write_skill(
        root=tmp_path,
        name="refund-policy",
        description="Handle refund-policy questions.",
        body="# Refund Policy\n\nUse for returns.",
    )
    (valid_root / "scripts").mkdir()
    (valid_root / "scripts" / "run.py").write_text("print('ok')", encoding="utf-8")
    (valid_root / "references").mkdir()
    (valid_root / "references" / "policy.md").write_text("Policy", encoding="utf-8")
    (valid_root / "notes.txt").write_text("Notes", encoding="utf-8")

    invalid_root = tmp_path / "broken-skill"
    invalid_root.mkdir()
    (invalid_root / "SKILL.md").write_text(
        "---\ndescription: Missing name\n---\n\nBroken",
        encoding="utf-8",
    )

    loader = FilesystemSkillLoader(
        roots=(tmp_path,),
        include_default_roots=False,
    )

    async def scenario() -> None:
        manifests = await loader.discover()
        assert [manifest.name for manifest in manifests] == ["refund-policy"]

        loaded = await loader.load("refund-policy")
        assert loaded.manifest.root == valid_root.resolve()
        assert loaded.resources == (
            SkillResource(
                path="scripts/run.py",
            ),
            SkillResource(
                path="references/policy.md",
            ),
            SkillResource(
                path="notes.txt",
            ),
        )

    asyncio.run(scenario())


def test_filesystem_skill_loader_uses_default_roots(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    home_root = tmp_path / "home"
    cwd_root = tmp_path / "workspace"
    cwd_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_root))
    monkeypatch.chdir(cwd_root)
    (cwd_root / ".agents" / "skills").mkdir(parents=True)
    (home_root / ".agents" / "skills").mkdir(parents=True)

    _write_skill(
        root=cwd_root / ".agents" / "skills",
        name="cwd-skill",
        description="Skill from cwd.",
        body="# CWD\n\nUse the cwd skill.",
    )
    _write_skill(
        root=home_root / ".agents" / "skills",
        name="home-skill",
        description="Skill from home.",
        body="# Home\n\nUse the home skill.",
    )

    loader = FilesystemSkillLoader()

    async def scenario() -> None:
        manifests = await loader.discover()
        assert [manifest.name for manifest in manifests] == [
            "cwd-skill",
            "home-skill",
        ]

    asyncio.run(scenario())


def test_skills_shim_activates_skill_from_custom_loader_and_deduplicates() -> None:
    loaded_skill = LoadedSkill(
        manifest=SkillManifest(
            name="refund-policy",
            description="Explain refund and return policy.",
            skill_file=Path("/skills/refund-policy/SKILL.md"),
            root=Path("/skills/refund-policy"),
        ),
        instructions="# Refund Policy\n\nUse this skill for refund questions.",
        resources=(
            SkillResource(
                path="references/policy.md",
            ),
        ),
    )
    loader = _MemorySkillLoader(loaded_skill)
    model = _SequenceModel(
        [
            _assistant_response(
                None,
                tool_calls=[
                    _make_tool_call(
                        tool_id="call_1",
                        name="activate_skill",
                        arguments=json.dumps({"name": "refund-policy"}),
                    )
                ],
            ),
            _assistant_response(
                None,
                tool_calls=[
                    _make_tool_call(
                        tool_id="call_2",
                        name="activate_skill",
                        arguments=json.dumps({"name": "refund-policy"}),
                    )
                ],
            ),
            _assistant_response("done"),
        ]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Support",
            model=model,
            instructions="You are a support assistant.",
            shims=(SkillsShim(loader=loader),),
        )
    )

    async def scenario() -> None:
        result = await agent.run("Can I still return my order?")

        assert result.final_output == "done"
        assert loader.discover_calls == 1
        assert loader.load_calls == ["refund-policy"]

        first_call_tools = model.call_options[0]["tools"]
        if not isinstance(first_call_tools, Tools):
            raise AssertionError("Expected skills tool registration.")
        assert [tool.name for tool in first_call_tools.normalized_tools] == [
            "activate_skill"
        ]
        assert first_call_tools.normalized_tools[0].args_type() is ActivateSkillInput

        assert model.calls[0][0]["role"] == "system"
        system_prompt = str(model.calls[0][0]["content"])
        assert system_prompt.startswith("You are a support assistant.\n\n")
        assert "<skills_system>" in system_prompt
        assert (
            "You have access to the following skills that provide specialized instructions for specific tasks."
            in system_prompt
        )
        assert (
            "When a task matches a skill's description, call the activate_skill tool"
            in system_prompt
        )
        assert "with the skill's name to load its full instructions." in system_prompt
        assert "<available_skills>" in system_prompt
        assert "<name>refund-policy</name>" in system_prompt
        assert (
            "<description>Explain refund and return policy.</description>"
            in system_prompt
        )
        assert "<location>/skills/refund-policy/SKILL.md</location>" in system_prompt
        assert "Typical workflow:" in system_prompt
        assert (
            'User: "A customer wants to know whether a damaged package still qualifies for a refund."'
            in system_prompt
        )
        assert (
            "1. Compare the user's request to the available skill descriptions."
            in system_prompt
        )
        assert (
            '2. Notice that the refund-policy skill matches and call activate_skill with "refund-policy".'
            in system_prompt
        )
        assert (
            "3. Read the returned <skill_content> instructions and follow that workflow."
            in system_prompt
        )
        assert (
            "4. Use any listed <skill_resources> paths relative to the skill directory when you need supporting files."
            in system_prompt
        )
        assert "</skills_system>" in system_prompt
        assert model.calls[0][1] == {
            "role": "user",
            "content": "Can I still return my order?",
        }

        assert model.calls[1][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "activate_skill",
            "content": (
                '<skill_content name="refund-policy">\n'
                "# Refund Policy\n\n"
                "Use this skill for refund questions.\n"
                "\n"
                "Skill directory: /skills/refund-policy\n"
                "Relative paths in this skill are relative to the skill directory.\n"
                "\n"
                "<skill_resources>\n"
                "  <file>references/policy.md</file>\n"
                "</skill_resources>\n"
                "</skill_content>"
            ),
        }
        assert model.calls[2][-1] == {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "activate_skill",
            "content": "Skill `refund-policy` is already active in this run.",
        }

        if agent.run_state is None:
            raise AssertionError("Expected persisted run state after completion.")
        assert isinstance(agent.run_state.shim_state, ShimState)
        assert agent.run_state.shim_state == {
            "skills:active-skill-names": ["refund-policy"],
        }

    asyncio.run(scenario())


def test_skills_shim_activation_is_visible_to_runner_hooks() -> None:
    loaded_skill = LoadedSkill(
        manifest=SkillManifest(
            name="refund-policy",
            description="Explain refund and return policy.",
            skill_file=Path("/skills/refund-policy/SKILL.md"),
            root=Path("/skills/refund-policy"),
        ),
        instructions="# Refund Policy\n\nUse this skill for refund questions.",
        resources=(),
    )
    loader = _MemorySkillLoader(loaded_skill)
    model = _SequenceModel(
        [
            _assistant_response(
                None,
                tool_calls=[
                    _make_tool_call(
                        tool_id="call_1",
                        name="activate_skill",
                        arguments=json.dumps({"name": "refund-policy"}),
                    )
                ],
            ),
            _assistant_response("done"),
        ]
    )
    hooks = _SkillActivationRecordingHooks()
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Support",
            model=model,
            instructions="You are a support assistant.",
            shims=(SkillsShim(loader=loader),),
        ),
        hooks=hooks,
    )

    async def scenario() -> None:
        result = await agent.run("Can I still return my order?")

        assert result.final_output == "done"
        assert hooks.events == [
            ("agent_start", "Support"),
            ("tool_start", '{"name": "refund-policy"}'),
            ("tool_end", "loaded"),
            ("agent_end", "done"),
        ]

    asyncio.run(scenario())


def test_skills_shim_uses_custom_system_prompt() -> None:
    loaded_skill = LoadedSkill(
        manifest=SkillManifest(
            name="refund-policy",
            description="Explain refund and return policy.",
            skill_file=Path("/skills/refund-policy/SKILL.md"),
            root=Path("/skills/refund-policy"),
        ),
        instructions="# Refund Policy",
        resources=(),
    )
    loader = _MemorySkillLoader(loaded_skill)
    model = _SequenceModel([_assistant_response("done")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Support",
            model=model,
            instructions="You are a support assistant.",
            shims=(
                SkillsShim(
                    loader=loader,
                    system_prompt="Use skills when they match the request.",
                ),
            ),
        )
    )

    async def scenario() -> None:
        result = await agent.run("Hello")

        assert result.final_output == "done"
        assert model.calls[0][0]["role"] == "system"
        assert str(model.calls[0][0]["content"]).startswith(
            "You are a support assistant.\n\nUse skills when they match the request."
        )

    asyncio.run(scenario())


def test_skills_shim_omits_prompt_and_tool_when_no_skills_are_discovered() -> None:
    model = _SequenceModel([_assistant_response("done")])
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Support",
            model=model,
            instructions="You are a support assistant.",
            shims=(SkillsShim(loader=_EmptySkillLoader()),),
        )
    )

    async def scenario() -> None:
        result = await agent.run("Hello")

        assert result.final_output == "done"
        assert model.calls[0] == [
            {
                "role": "system",
                "content": "You are a support assistant.",
            },
            {
                "role": "user",
                "content": "Hello",
            },
        ]
        assert model.call_options[0]["tools"] is None

    asyncio.run(scenario())


def _make_manifest(name: str, description: str = "Test skill.") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        skill_file=Path(f"/skills/{name}/SKILL.md"),
        root=Path(f"/skills/{name}"),
    )


def test_skill_catalog_get_returns_manifest_for_known_name() -> None:
    manifest = _make_manifest("refund-policy")
    catalog = SkillCatalog(manifests=[manifest], loader=_EmptySkillLoader())

    assert catalog.get("refund-policy") is manifest


def test_skill_catalog_get_returns_none_for_unknown_name() -> None:
    manifest = _make_manifest("refund-policy")
    catalog = SkillCatalog(manifests=[manifest], loader=_EmptySkillLoader())

    assert catalog.get("unknown") is None


def test_skill_catalog_has_returns_true_for_known_and_false_for_unknown() -> None:
    manifest = _make_manifest("refund-policy")
    catalog = SkillCatalog(manifests=[manifest], loader=_EmptySkillLoader())

    assert catalog.has("refund-policy") is True
    assert catalog.has("unknown") is False


def test_skill_catalog_names_returns_stable_order() -> None:
    manifests = [
        _make_manifest("alpha"),
        _make_manifest("beta"),
        _make_manifest("gamma"),
    ]
    catalog = SkillCatalog(manifests=manifests, loader=_EmptySkillLoader())

    assert catalog.names() == ("alpha", "beta", "gamma")


def test_skill_catalog_iter_and_len() -> None:
    manifests = [_make_manifest("a"), _make_manifest("b")]
    catalog = SkillCatalog(manifests=manifests, loader=_EmptySkillLoader())

    assert len(catalog) == 2
    assert list(catalog) == manifests


def test_skill_catalog_empty() -> None:
    catalog = SkillCatalog(manifests=[], loader=_EmptySkillLoader())

    assert len(catalog) == 0
    assert catalog.names() == ()
    assert list(catalog) == []
    assert catalog.get("anything") is None
    assert catalog.has("anything") is False


def test_skill_catalog_load_raises_key_error_for_unknown_name() -> None:
    manifest = _make_manifest("refund-policy")
    catalog = SkillCatalog(manifests=[manifest], loader=_EmptySkillLoader())

    with pytest.raises(KeyError):
        asyncio.run(catalog.load("unknown"))


def test_skill_catalog_load_delegates_to_loader() -> None:
    loaded_skill = LoadedSkill(
        manifest=_make_manifest("refund-policy"),
        instructions="# Refund Policy",
        resources=(),
    )
    loader = _MemorySkillLoader(loaded_skill)
    catalog = SkillCatalog(manifests=[loaded_skill.manifest], loader=loader)

    result = asyncio.run(catalog.load("refund-policy"))

    assert result is loaded_skill
    assert loader.load_calls == ["refund-policy"]


# ---------------------------------------------------------------------------
# Prompt rendering unit tests
# ---------------------------------------------------------------------------


def test_render_skills_system_prompt_includes_skill_metadata() -> None:
    manifests = (
        _make_manifest("refund-policy", "Handle refund questions."),
        _make_manifest("shipping-info", "Provide shipping details."),
    )
    context = SkillsSystemPromptContext(tool_name="activate_skill", skills=manifests)
    result = render_skills_system_prompt(
        template=(
            "<skills>{% for skill in skills %}"
            "<skill><name>{{ skill.name }}</name></skill>"
            "{% endfor %}</skills>"
        ),
        context=context,
    )

    assert "<name>refund-policy</name>" in result
    assert "<name>shipping-info</name>" in result


def test_render_loaded_skill_includes_instructions_and_resources() -> None:
    loaded_skill = LoadedSkill(
        manifest=_make_manifest("refund-policy"),
        instructions="# Refund Policy\n\nFollow these steps.",
        resources=(
            SkillResource(path="scripts/run.py"),
            SkillResource(path="references/policy.md"),
        ),
    )
    result = render_loaded_skill(loaded_skill)

    assert '<skill_content name="refund-policy">' in result
    assert "# Refund Policy" in result
    assert "Follow these steps." in result
    assert "Skill directory: /skills/refund-policy" in result
    assert "<file>scripts/run.py</file>" in result
    assert "<file>references/policy.md</file>" in result
    assert "</skill_content>" in result


def test_render_loaded_skill_with_empty_resources() -> None:
    loaded_skill = LoadedSkill(
        manifest=_make_manifest("simple-skill"),
        instructions="Just do the thing.",
        resources=(),
    )
    result = render_loaded_skill(loaded_skill)

    assert '<skill_content name="simple-skill">' in result
    assert "Just do the thing." in result
    assert "<skill_resources>" in result


def test_filesystem_skill_loader_load_raises_key_error_for_unknown_name(
    tmp_path: Path,
) -> None:
    _write_skill(
        root=tmp_path,
        name="refund-policy",
        description="Handle refund questions.",
        body="# Refund Policy",
    )
    loader = FilesystemSkillLoader(roots=(tmp_path,), include_default_roots=False)

    async def scenario() -> None:
        await loader.discover()
        with pytest.raises(KeyError):
            await loader.load("nonexistent-skill")

    asyncio.run(scenario())


def test_filesystem_skill_loader_load_uses_cache_from_discover(
    tmp_path: Path,
) -> None:
    _write_skill(
        root=tmp_path,
        name="refund-policy",
        description="Handle refund questions.",
        body="# Refund Policy",
    )
    loader = FilesystemSkillLoader(roots=(tmp_path,), include_default_roots=False)

    async def scenario() -> None:
        manifests = await loader.discover()
        assert len(manifests) == 1

        loaded = await loader.load("refund-policy")
        assert loaded.manifest.name == "refund-policy"
        assert loaded.instructions == "# Refund Policy"

    asyncio.run(scenario())


class _MultiSkillLoader(SkillLoader):
    """Loader that exposes multiple skills for multi-activation tests."""

    def __init__(self, skills: list[LoadedSkill]) -> None:
        self._skills = {skill.manifest.name: skill for skill in skills}
        self.load_calls: list[str] = []

    async def discover(self) -> tuple[SkillManifest, ...]:
        return tuple(skill.manifest for skill in self._skills.values())

    async def load(self, name: str) -> LoadedSkill:
        self.load_calls.append(name)
        if name not in self._skills:
            raise KeyError(name)
        return self._skills[name]


def test_skills_shim_activates_multiple_different_skills_in_one_run() -> None:
    skill_a = LoadedSkill(
        manifest=SkillManifest(
            name="refund-policy",
            description="Handle refunds.",
            skill_file=Path("/skills/refund-policy/SKILL.md"),
            root=Path("/skills/refund-policy"),
        ),
        instructions="# Refund instructions",
        resources=(),
    )
    skill_b = LoadedSkill(
        manifest=SkillManifest(
            name="shipping-info",
            description="Handle shipping.",
            skill_file=Path("/skills/shipping-info/SKILL.md"),
            root=Path("/skills/shipping-info"),
        ),
        instructions="# Shipping instructions",
        resources=(),
    )
    loader = _MultiSkillLoader([skill_a, skill_b])
    model = _SequenceModel(
        [
            _assistant_response(
                None,
                tool_calls=[
                    _make_tool_call(
                        tool_id="call_1",
                        name="activate_skill",
                        arguments=json.dumps({"name": "refund-policy"}),
                    )
                ],
            ),
            _assistant_response(
                None,
                tool_calls=[
                    _make_tool_call(
                        tool_id="call_2",
                        name="activate_skill",
                        arguments=json.dumps({"name": "shipping-info"}),
                    )
                ],
            ),
            _assistant_response("done"),
        ]
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Support",
            model=model,
            instructions="You are a support assistant.",
            shims=(SkillsShim(loader=loader),),
        )
    )

    async def scenario() -> None:
        result = await agent.run("Help me with my order")

        assert result.final_output == "done"
        assert loader.load_calls == ["refund-policy", "shipping-info"]

        first_activation = model.calls[1][-1]
        assert first_activation["name"] == "activate_skill"
        assert "# Refund instructions" in str(first_activation["content"])

        second_activation = model.calls[2][-1]
        assert second_activation["name"] == "activate_skill"
        assert "# Shipping instructions" in str(second_activation["content"])

        if agent.run_state is None:
            raise AssertionError("Expected persisted run state after completion.")
        assert isinstance(agent.run_state.shim_state, ShimState)
        assert agent.run_state.shim_state == {
            "skills:active-skill-names": ["refund-policy", "shipping-info"],
        }

    asyncio.run(scenario())
