"""Real OpenAI-backed patient-care demo with templated instructions and resume."""

import asyncio
import logging
import os
from typing import TypedDict

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, OutputSchema, PromptSpec, PromptTemplate

MODEL_NAME = "gpt-5.4-mini"


class CareInstructionValues(TypedDict):
    """Typed values used to render the care-navigation instructions."""

    clinic_name: str
    nurse_line_hours: str
    urgent_symptoms: str
    escalation_channel: str
    communication_style: str


class PatientMessageValues(TypedDict):
    """Typed values used to render the initial patient message."""

    clinic_name: str
    patient_label: str
    symptom_summary: str
    timing: str


# The descriptor instructions are system-only in this example, so the prompt
# template does not need to manufacture a fake user message anymore.
CARE_INSTRUCTIONS_TEMPLATE = PromptTemplate[CareInstructionValues, str](
    system_template="""
You are {{ clinic_name }}'s patient-care navigation assistant.
Use a {{ communication_style }} tone and keep replies concise, practical, and honest.

Care snapshot:
- Nurse line hours: {{ nurse_line_hours }}
- Escalate urgently for: {{ urgent_symptoms }}
- Clinician escalation channel: {{ escalation_channel }}

Rules:
- Never diagnose, prescribe, or claim you checked the chart.
- Ask for missing details before suggesting a resolution.
- If enough details are available, explain the likely care-team next step clearly.
- Keep replies under 120 words unless the patient explicitly asks for more detail.
""".strip(),
    output_schema=OutputSchema(str),
)

INITIAL_PATIENT_MESSAGE_TEMPLATE = PromptTemplate[PatientMessageValues, str](
    system_template=None,
    user_template="""
Hi, I need help from {{ clinic_name }}.
Patient: {{ patient_label }}.
Concern: {{ symptom_summary }}.
Timing: {{ timing }}.
    """.strip(),
    output_schema=OutputSchema(str),
)


async def run_demo() -> None:
    """Run the real multi-turn patient-care demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    instruction_values: CareInstructionValues = {
        "clinic_name": "Harborview Endocrinology",
        "nurse_line_hours": "weekdays 8:00-17:00",
        "urgent_symptoms": "confusion, fainting, chest pain, severe weakness, or glucose under 54 mg/dL",
        "escalation_channel": "endocrine-triage@harborview.test",
        "communication_style": "warm and clinically careful",
    }
    descriptor = AgentDescriptor(
        name="Patient Care Navigator",
        description="Frontline patient-care navigation assistant",
        model=model,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=CARE_INSTRUCTIONS_TEMPLATE,
            values=instruction_values,
        ),
    )

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=descriptor,
    )

    # This demo currently simulates patient turns by sending them directly from the
    # script. That keeps the example focused on the harness surface:
    # instructions, multi-turn continuation, and `RunState` resume. It is not a
    # proper clinician handoff flow yet.
    initial_patient_values: PatientMessageValues = {
        "clinic_name": "Harborview Endocrinology",
        "patient_label": "Maya R., 54F",
        "symptom_summary": "my glucose has been in the 60s twice since starting a new injection",
        "timing": "the low readings happened this morning and after lunch",
    }
    initial_patient_text = (
        "Hi, I need help from Harborview Endocrinology. Patient: Maya R., 54F. "
        "Concern: my glucose has been in the 60s twice since starting a new injection. "
        "Timing: the low readings happened this morning and after lunch."
    )
    print("Example: patient care conversation with resume")
    print(f"Model: {MODEL_NAME}")
    print()
    print(
        "Note: this demo simulates patient follow-up turns by sending them from the script."
    )
    print("It does not implement proper clinician handoff yet.")
    print()

    first_result = await agent.run(
        [
            PromptSpec(
                template=INITIAL_PATIENT_MESSAGE_TEMPLATE,
                values=initial_patient_values,
            )
        ]
    )
    print(f"Patient: {initial_patient_text}")
    print(f"Assistant: {first_result.final_output}")
    print()

    second_patient_text = (
        "Thanks. I felt shaky but did not faint, and I took glucose tablets. "
        "My current reading is 92."
    )
    second_result = await agent.run(second_patient_text)
    print(f"Patient: {second_patient_text}")
    print(f"Assistant: {second_result.final_output}")
    print()

    saved_run_state = agent.run_state
    if saved_run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print(
        "Simulated restart:"
        f" saved RunState after {saved_run_state.turn_count} turns and"
        f" {len(saved_run_state.responses)} raw model responses."
    )
    print()

    resumed_agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=descriptor,
        run_state=saved_run_state,
    )

    third_patient_text = (
        "Please summarize the next steps in 3 bullet points so I know what to do next."
    )
    resumed_result = await resumed_agent.run(third_patient_text)
    print(f"Patient: {third_patient_text}")
    print(f"Assistant: {resumed_result.final_output}")
    print()

    resumed_agent_state = resumed_agent.run_state
    if resumed_agent_state is None:
        raise RuntimeError("Expected the resumed default agent to persist run state.")

    print(
        "Run summary:"
        f" resumed successfully, {resumed_agent_state.turn_count} total turns,"
        f" {len(resumed_agent_state.responses)} raw responses."
    )


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
