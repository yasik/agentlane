"""Real OpenAI-backed streaming example with tool call, agent-as-tool, and handoff."""

import asyncio
import logging
import os
from typing import cast

import structlog
from agentlane_openai import ResponsesClient
from pydantic import BaseModel, Field

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import (
    Config,
    ModelResponse,
    ModelStreamEventKind,
    ToolCall,
    Tools,
    as_tool,
)

MODEL_NAME = "gpt-5.4-mini"

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


@as_tool
async def lookup_patient_snapshot(patient_id: str) -> str:
    """Look up the current mock chart facts for one patient."""
    del patient_id
    return (
        "Patient MR-4421: 54F with type 2 diabetes started semaglutide this week. "
        "She reports dizziness and two glucose readings in the 60s today. No "
        "syncope reported yet. Current reading after glucose tablets is 92."
    )


class MedicationSafetyArgs(BaseModel):
    """Arguments exposed for the medication-safety agent tool."""

    patient_id: str = Field(description="The patient id being reviewed.")
    medication_name: str = Field(description="The medication being reviewed.")
    issue_summary: str = Field(
        description="A short summary of the symptom or medication-safety issue."
    )


NURSE_TRIAGE_SPECIALIST = AgentDescriptor(
    name="Nurse Triage Specialist",
    description="Takes over patient conversations that need clinical escalation.",
    model=MODEL,
    model_args={"reasoning": {"effort": "low", "summary": "detailed"}},
    tools=None,
    instructions=(
        "You are the nurse triage specialist. "
        "The conversation has already been triaged and includes chart facts plus "
        "medication-safety findings. Take over directly with the patient. "
        "Do not diagnose or change medications. "
        "Explain the next step in at most three bullet points."
    ),
)

MEDICATION_SAFETY_SPECIALIST = AgentDescriptor(
    name="Medication Safety Specialist",
    description="Reviews medication-safety concerns for a specific patient message.",
    model=MODEL,
    model_args={"reasoning": {"effort": "low", "summary": "detailed"}},
    tools=None,
    instructions=(
        "You are the medication safety specialist. "
        "Safety policy: recurrent glucose readings under 70 mg/dL after a new "
        "diabetes medication require same-day clinician review; readings under "
        "54 mg/dL, confusion, fainting, or chest pain require urgent escalation. "
        "Return one short sentence with the safety ruling."
    ),
)


class FrontlineCareAgent(DefaultAgent):
    """Frontline care agent that gathers facts, consults safety, then transfers."""

    descriptor = AgentDescriptor(
        name="Frontline Care Navigator",
        description="Frontline care navigation that escalates medication-safety issues.",
        model=MODEL,
        model_args={"reasoning": {"effort": "medium", "summary": "detailed"}},
        instructions=(
            "You are frontline patient care navigation. "
            "For low-glucose medication-safety questions, follow this exact sequence: "
            "1. In one short two-step preamble, explain what you are about to verify. "
            "2. Call `lookup_patient_snapshot` exactly once. "
            "3. Call `medication_safety_specialist` exactly once with the chart facts "
            "and issue summary. "
            "4. If the case needs same-day clinician review or urgent escalation, "
            "handoff to `nurse_triage_specialist`. "
            "Do not handoff before steps 2 and 3 are complete."
        ),
        tools=Tools(
            tools=[
                lookup_patient_snapshot,
                MEDICATION_SAFETY_SPECIALIST.as_tool(
                    name="medication_safety_specialist",
                    description="Ask the medication-safety specialist for a final safety ruling.",
                    args_model=MedicationSafetyArgs,
                ),
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            tool_call_limits={
                "lookup_patient_snapshot": 1,
                "medication_safety_specialist": 1,
            },
        ),
        handoffs=(NURSE_TRIAGE_SPECIALIST,),
    )


def _tool_path(run_state_responses: list[ModelResponse]) -> list[str]:
    """Extract the sequence of tool or handoff calls from raw responses."""
    names: list[str] = []
    for response in run_state_responses:
        tool_calls = cast(list[ToolCall], response.choices[0].message.tool_calls or [])
        for tool_call in tool_calls:
            name = tool_call.function.name or ""
            if name:
                names.append(name)
    return names


async def run_demo() -> None:
    """Run the orchestration streaming example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = FrontlineCareAgent()
    user_message = (
        "I started my new diabetes injection and my glucose was 64 this morning "
        "and 68 after lunch. I felt shaky. What should I do?"
    )
    stream = await agent.run_stream(user_message)

    print("Example: streaming escalation flow")
    print(f"Model: {MODEL_NAME}")
    print()
    print("One streamed run exercises:")
    print("1. a normal tool call")
    print("2. a predefined agent-as-tool delegation")
    print("3. a first-class handoff that keeps streaming through the specialist")
    print()
    print(f"User: {user_message}")
    print("Assistant stream:")

    last_openai_phase: str | None = None
    saw_reasoning = False
    async for event in stream:
        if event.kind == ModelStreamEventKind.REASONING and event.reasoning is not None:
            reasoning_value = event.reasoning
            reasoning_text = (
                reasoning_value
                if isinstance(reasoning_value, str)
                else str(reasoning_value)
            )
            if reasoning_text.strip():
                if not saw_reasoning:
                    print()
                    print("[reasoning]")
                    saw_reasoning = True
                print(reasoning_text, end="", flush=True)
            continue

        if (
            event.kind == ModelStreamEventKind.PROVIDER
            and event.provider_event_type is not None
        ):
            raw_item = getattr(event.raw, "item", None)
            raw_phase = getattr(raw_item, "phase", None)
            if isinstance(raw_phase, str) and raw_phase != last_openai_phase:
                last_openai_phase = raw_phase
                print()
                print(f"[openai phase] {raw_phase}")
            continue

        if event.kind == ModelStreamEventKind.TEXT_DELTA and event.text:
            print(event.text, end="", flush=True)
            continue

        if (
            event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA
            and event.arguments_delta
        ):
            print()
            print(f"[tool arguments] {event.arguments_delta}")

    result = await stream.result()
    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the streamed run to persist run state.")

    print()
    print()
    print(f"Final output: {result.final_output}")
    print(f"Tool and handoff path: {' -> '.join(_tool_path(run_state.responses))}")
    print(
        "Run summary:"
        f" {run_state.turn_count} turns,"
        f" {len(run_state.responses)} raw model responses."
    )
    print()
    print(
        "Note: the outer stream shows the parent run plus the transferred "
        "specialist run after handoff. The internal medication-safety-specialist "
        "agent-as-tool run stays internal in the current streaming contract."
    )


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
