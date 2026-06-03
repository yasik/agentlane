"""Runtime agents for the distributed clinical inbox demo."""

from __future__ import annotations

import asyncio
from datetime import datetime

from agentlane.messaging import CorrelationId, DeliveryMode, TopicId
from agentlane.runtime import (
    BaseAgent,
    Engine,
    MessageContext,
    WorkerAgentRuntime,
    on_message,
)
from examples.harness.distributed_clinical_inbox_copilot.messages import (
    AGGREGATOR_AGENT_TYPE,
    CHART_HISTORY_AGENT_TYPE,
    GUIDELINE_AGENT_TYPE,
    MED_SAFETY_AGENT_TYPE,
    PATIENT_COMMS_AGENT_TYPE,
    RESULT_TOPIC_TYPE,
    REVIEW_COMPLETED_TOPIC_TYPE,
    REVIEW_TOPIC_TYPE,
    SPECIALIST_NAMES,
    TELEMETRY_TOPIC_TYPE,
    WORKER_ROLE_TO_AGENT_TYPE,
    AggregatedClinicalReview,
    ClinicalReviewTask,
    DemoEvent,
    SpecialistFinding,
)


async def publish_demo_event(
    agent: BaseAgent,
    *,
    session_id: str,
    actor: str,
    message: str,
    status_target: str | None = None,
    status_state: str | None = None,
    status_detail: str | None = None,
    correlation_id: CorrelationId | None = None,
) -> None:
    """Publish one UI event through the distributed runtime."""
    await agent.publish_message(
        DemoEvent(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            actor=actor,
            message=message,
            status_target=status_target,
            status_state=status_state,
            status_detail=status_detail,
        ),
        topic=TopicId.from_values(
            type_value=TELEMETRY_TOPIC_TYPE,
            route_key=session_id,
        ),
        correlation_id=correlation_id,
    )


class MedSafetyAgent(BaseAgent):
    """Specialist that reviews likely medication-safety signals."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the specialist agent."""
        super().__init__(engine)

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Review medication-safety concerns and publish one finding."""
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="med-safety-agent",
            message=(
                "│ med-safety-agent      checking for medication-related hypoglycemia"
            ),
            status_target="med-safety-agent",
            status_state="running",
            status_detail="Reviewing medication safety signals",
            correlation_id=context.correlation_id,
        )
        await asyncio.sleep(0.35)
        finding = SpecialistFinding(
            session_id=payload.session_id,
            review_id=payload.review_id,
            agent_name="med-safety-agent",
            headline="Medication safety",
            detail=(
                "The reported dizziness plus glucose values under 70 raise concern "
                "for symptomatic hypoglycemia. The chart pattern makes glipizide the "
                "most likely immediate contributor to low readings after the recent "
                "regimen change."
            ),
            patient_reply_guidance=(
                "Tell the patient not to make unsupervised medication changes, but to "
                "treat any low sugar per their plan and wait for a same-day clinician "
                "reply."
            ),
            urgent_flag=True,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="med-safety-agent",
            message="└ med-safety-agent      flagged hypoglycemia risk and likely culprit",
            status_target="med-safety-agent",
            status_state="done",
            status_detail="Flagged symptomatic hypoglycemia risk",
            correlation_id=context.correlation_id,
        )
        return finding


class GuidelineAgent(BaseAgent):
    """Specialist that summarizes the relevant care-path guidance."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the specialist agent."""
        super().__init__(engine)

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Review the message against guideline-like escalation logic."""
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="guideline-agent",
            message="│ guideline-agent       reviewing escalation guidance for low glucose",
            status_target="guideline-agent",
            status_state="running",
            status_detail="Matching guideline-style escalation signals",
            correlation_id=context.correlation_id,
        )
        await asyncio.sleep(0.2)
        finding = SpecialistFinding(
            session_id=payload.session_id,
            review_id=payload.review_id,
            agent_name="guideline-agent",
            headline="Guideline alignment",
            detail=(
                "Recurrent symptomatic glucose values below 70 should trigger "
                "same-day clinical review and clear escalation instructions if "
                "symptoms worsen, the patient cannot keep glucose above target, "
                "or new confusion, syncope, or chest pain appears."
            ),
            patient_reply_guidance=(
                "Include explicit red-flag instructions for severe symptoms or "
                "persistent lows that do not improve quickly."
            ),
            urgent_flag=True,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="guideline-agent",
            message="└ guideline-agent       recommended same-day clinician review",
            status_target="guideline-agent",
            status_state="done",
            status_detail="Recommended same-day clinical review",
            correlation_id=context.correlation_id,
        )
        return finding


class ChartHistoryAgent(BaseAgent):
    """Specialist that extracts relevant chart context."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the specialist agent."""
        super().__init__(engine)

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Summarize the most relevant chart history for the inbox message."""
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="chart-history-agent",
            message="│ chart-history-agent   extracting recent labs, meds, and symptoms",
            status_target="chart-history-agent",
            status_state="running",
            status_detail="Pulling chart context",
            correlation_id=context.correlation_id,
        )
        await asyncio.sleep(0.28)
        finding = SpecialistFinding(
            session_id=payload.session_id,
            review_id=payload.review_id,
            agent_name="chart-history-agent",
            headline="Chart context",
            detail=(
                f"The chart snapshot for {payload.patient_label} shows type 2 diabetes, "
                "glipizide plus metformin, and semaglutide started two weeks ago. Recent "
                "home readings in the low 60s line up with the patient message and there "
                "is no ED visit or severe-event documentation in the mock chart."
            ),
            patient_reply_guidance=(
                "Ask the patient to confirm their latest glucose reading and whether the "
                "dizziness is improving after eating or treating the low."
            ),
            urgent_flag=False,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="chart-history-agent",
            message=(
                "└ chart-history-agent   summarized the mock chart and recent glucose trend"
            ),
            status_target="chart-history-agent",
            status_state="done",
            status_detail="Summarized chart context",
            correlation_id=context.correlation_id,
        )
        return finding


class PatientCommsAgent(BaseAgent):
    """Specialist that suggests patient-friendly messaging."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the specialist agent."""
        super().__init__(engine)

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Draft plain-language patient communication guidance."""
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="patient-comms-agent",
            message="│ patient-comms-agent   drafting a plain-language patient response",
            status_target="patient-comms-agent",
            status_state="running",
            status_detail="Drafting patient-friendly guidance",
            correlation_id=context.correlation_id,
        )
        await asyncio.sleep(0.32)
        finding = SpecialistFinding(
            session_id=payload.session_id,
            review_id=payload.review_id,
            agent_name="patient-comms-agent",
            headline="Patient communication",
            detail=(
                "The response should acknowledge the symptoms, advise the patient to "
                "treat any low sugar per their plan, and set expectations that the care "
                "team will review medication safety the same day."
            ),
            patient_reply_guidance=(
                "Use calm plain language, ask about the current glucose value, and tell "
                "the patient to seek urgent help for severe weakness, confusion, fainting, "
                "or symptoms that do not improve."
            ),
            urgent_flag=False,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="patient-comms-agent",
            message="└ patient-comms-agent   prepared patient-friendly response guidance",
            status_target="patient-comms-agent",
            status_state="done",
            status_detail="Prepared patient response guidance",
            correlation_id=context.correlation_id,
        )
        return finding


class ReviewAggregatorAgent(BaseAgent):
    """Stateful aggregator keyed by review id."""

    def __init__(self, engine: Engine, expected_finding_count: int) -> None:
        """Initialize the aggregator dependencies."""
        super().__init__(engine)
        self._expected_finding_count = expected_finding_count
        self._findings: dict[str, SpecialistFinding] = {}

    @on_message
    async def handle(
        self,
        payload: SpecialistFinding,
        context: MessageContext,
    ) -> object:
        """Collect specialist findings and publish the aggregate review."""
        self._findings[payload.agent_name] = payload
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="aggregator",
            message=(
                "→ aggregator collected "
                f"{len(self._findings)}/{self._expected_finding_count} findings"
            ),
            correlation_id=context.correlation_id,
        )
        if len(self._findings) < self._expected_finding_count:
            return None

        ordered_findings = {
            name: self._findings[name] for name in sorted(self._findings)
        }
        review = AggregatedClinicalReview(
            session_id=payload.session_id,
            review_id=self.id.key.value,
            findings={
                name: finding_to_payload(finding)
                for name, finding in ordered_findings.items()
            },
            urgent_flag=any(item.urgent_flag for item in ordered_findings.values()),
        )
        await publish_demo_event(
            self,
            session_id=payload.session_id,
            actor="aggregator",
            message=f"→ review {review.review_id} merged into one summary",
            correlation_id=context.correlation_id,
        )
        await self.publish_message(
            review,
            topic=TopicId.from_values(
                type_value=REVIEW_COMPLETED_TOPIC_TYPE,
                route_key=payload.session_id,
            ),
            correlation_id=context.correlation_id,
        )
        return None


def register_worker_role(worker: WorkerAgentRuntime, role: str) -> None:
    """Register one subprocess or in-process worker role."""
    if role == "med-safety":
        worker.register_factory(MED_SAFETY_AGENT_TYPE, MedSafetyAgent)
        worker.subscribe_exact(
            topic_type=REVIEW_TOPIC_TYPE,
            agent_type=MED_SAFETY_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATELESS,
        )
        return
    if role == "guideline":
        worker.register_factory(GUIDELINE_AGENT_TYPE, GuidelineAgent)
        worker.subscribe_exact(
            topic_type=REVIEW_TOPIC_TYPE,
            agent_type=GUIDELINE_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATELESS,
        )
        return
    if role == "chart-history":
        worker.register_factory(CHART_HISTORY_AGENT_TYPE, ChartHistoryAgent)
        worker.subscribe_exact(
            topic_type=REVIEW_TOPIC_TYPE,
            agent_type=CHART_HISTORY_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATELESS,
        )
        return
    if role == "patient-comms":
        worker.register_factory(PATIENT_COMMS_AGENT_TYPE, PatientCommsAgent)
        worker.subscribe_exact(
            topic_type=REVIEW_TOPIC_TYPE,
            agent_type=PATIENT_COMMS_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATELESS,
        )
        return
    if role == "aggregator":
        worker.register_factory(
            AGGREGATOR_AGENT_TYPE,
            lambda engine: ReviewAggregatorAgent(
                engine,
                expected_finding_count=len(SPECIALIST_NAMES),
            ),
        )
        worker.subscribe_exact(
            topic_type=RESULT_TOPIC_TYPE,
            agent_type=AGGREGATOR_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATEFUL,
        )
        return

    valid_roles = ", ".join(sorted(WORKER_ROLE_TO_AGENT_TYPE))
    raise ValueError(f"Unknown worker role '{role}'. Expected one of: {valid_roles}.")


def finding_to_payload(finding: SpecialistFinding) -> dict[str, object]:
    """Return a JSON-shaped finding payload for aggregate review messages."""
    return {
        "session_id": finding.session_id,
        "review_id": finding.review_id,
        "agent_name": finding.agent_name,
        "headline": finding.headline,
        "detail": finding.detail,
        "patient_reply_guidance": finding.patient_reply_guidance,
        "urgent_flag": finding.urgent_flag,
    }
