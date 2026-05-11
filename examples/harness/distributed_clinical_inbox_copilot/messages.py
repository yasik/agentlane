"""Shared message types for the distributed clinical inbox demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

MODEL_NAME = "gpt-5.4-mini"

MED_SAFETY_AGENT_TYPE = "demo.clinical.med_safety"
GUIDELINE_AGENT_TYPE = "demo.clinical.guideline"
CHART_HISTORY_AGENT_TYPE = "demo.clinical.chart_history"
PATIENT_COMMS_AGENT_TYPE = "demo.clinical.patient_comms"
AGGREGATOR_AGENT_TYPE = "demo.clinical.aggregator"
TELEMETRY_AGENT_TYPE = "demo.clinical.telemetry_sink"
REVIEW_RESULT_AGENT_TYPE = "demo.clinical.review_result_sink"

REVIEW_TOPIC_TYPE = "demo.clinical.review_requested"
RESULT_TOPIC_TYPE = "demo.clinical.review_result"
TELEMETRY_TOPIC_TYPE = "demo.clinical.telemetry"
REVIEW_COMPLETED_TOPIC_TYPE = "demo.clinical.review_completed"

SPECIALIST_NAMES = (
    "med-safety-agent",
    "guideline-agent",
    "chart-history-agent",
    "patient-comms-agent",
)
SPECIALIST_WORKER_LABELS = {
    "med-safety-agent": "med-safety-worker",
    "guideline-agent": "guideline-worker",
    "chart-history-agent": "chart-history-worker",
    "patient-comms-agent": "patient-comms-worker",
}
SPECIALIST_AGENT_TYPES = {
    "med-safety-agent": MED_SAFETY_AGENT_TYPE,
    "guideline-agent": GUIDELINE_AGENT_TYPE,
    "chart-history-agent": CHART_HISTORY_AGENT_TYPE,
    "patient-comms-agent": PATIENT_COMMS_AGENT_TYPE,
}

WORKER_ROLE_ORDER = (
    "med-safety",
    "guideline",
    "chart-history",
    "patient-comms",
    "aggregator",
)
WORKER_ROLE_TO_LABEL = {
    "med-safety": "med-safety-worker",
    "guideline": "guideline-worker",
    "chart-history": "chart-history-worker",
    "patient-comms": "patient-comms-worker",
    "aggregator": "aggregator-worker",
}
WORKER_ROLE_TO_AGENT_TYPE = {
    "med-safety": MED_SAFETY_AGENT_TYPE,
    "guideline": GUIDELINE_AGENT_TYPE,
    "chart-history": CHART_HISTORY_AGENT_TYPE,
    "patient-comms": PATIENT_COMMS_AGENT_TYPE,
    "aggregator": AGGREGATOR_AGENT_TYPE,
}

TOPOLOGY_NODE_ORDER = (
    "host",
    "copilot-worker",
    "med-safety-worker",
    "guideline-worker",
    "chart-history-worker",
    "patient-comms-worker",
    "aggregator-worker",
)
TOPOLOGY_NODE_SPECS = {
    "host": ("control plane", ("routing", "ownership", "subscriptions")),
    "copilot-worker": ("streamed harness run", ("Clinical Inbox Copilot",)),
    "med-safety-worker": ("specialist worker", (MED_SAFETY_AGENT_TYPE,)),
    "guideline-worker": ("specialist worker", (GUIDELINE_AGENT_TYPE,)),
    "chart-history-worker": ("specialist worker", (CHART_HISTORY_AGENT_TYPE,)),
    "patient-comms-worker": ("specialist worker", (PATIENT_COMMS_AGENT_TYPE,)),
    "aggregator-worker": ("stateful fan-in", (AGGREGATOR_AGENT_TYPE,)),
}


class MessageTypeRegistrar(Protocol):
    """Minimal runtime surface needed for message type registration."""

    def register_message_type(self, message_type: type[object]) -> None:
        """Register one typed message payload."""


@dataclass(slots=True, frozen=True)
class DemoInputs:
    """Interactive inputs gathered from the clinician."""

    clinician_name: str
    patient_label: str
    patient_message: str


@dataclass(slots=True)
class ClinicalReviewTask:
    """Publish payload delivered to each specialist reviewer."""

    session_id: str
    review_id: str
    clinician_name: str
    patient_label: str
    patient_message: str
    chart_snapshot: str


@dataclass(slots=True)
class SpecialistFinding:
    """One specialist output published back to the aggregator."""

    session_id: str
    review_id: str
    agent_name: str
    headline: str
    detail: str
    patient_reply_guidance: str
    urgent_flag: bool


@dataclass(slots=True)
class AggregatedClinicalReview:
    """Merged specialist review for one inbox message."""

    session_id: str
    review_id: str
    findings: dict[str, dict[str, object]]
    urgent_flag: bool


@dataclass(slots=True, frozen=True)
class DemoEvent:
    """One structured UI event emitted by hooks or runtime agents."""

    timestamp: str
    actor: str
    message: str
    status_target: str | None = None
    status_state: str | None = None
    status_detail: str | None = None


@dataclass(slots=True)
class RuntimeNode:
    """One distributed runtime node shown in the live topology panel."""

    label: str
    role: str
    agent_types: tuple[str, ...]
    address: str = "pending"
    worker_id: str = "pending"
    pid: str = "pending"
    state: str = "starting"


def build_topology_nodes() -> dict[str, RuntimeNode]:
    """Build the initial topology metadata before sockets are bound."""
    return {
        label: RuntimeNode(
            label=label,
            role=TOPOLOGY_NODE_SPECS[label][0],
            agent_types=TOPOLOGY_NODE_SPECS[label][1],
        )
        for label in TOPOLOGY_NODE_ORDER
    }


def register_demo_message_types(registrar: MessageTypeRegistrar) -> None:
    """Register typed payloads that cross distributed worker boundaries."""
    registrar.register_message_type(ClinicalReviewTask)
    registrar.register_message_type(SpecialistFinding)
    registrar.register_message_type(AggregatedClinicalReview)
    registrar.register_message_type(DemoEvent)
