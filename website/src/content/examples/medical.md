---
id: medical
label: Medical
filename: clinic.example
note: synthetic case · illustrative activity
duration: 28000
channels:
  - id: portal
    label: patient portal
  - id: records
    label: records / labs
  - id: doctor
    label: doctor workspace
roles:
  - id: human
    name: physician
    duty: owns clinical decisions, revisions, and approval
    status: seeing patients
    children:
      - id: lead
        name: clinic.lead
        duty: owns case synthesis, evidence, and escalation
        status: overseeing active cases
        children:
          - id: intake
            name: clinic.intake
            duty: owns history, records, and missing information
            status: available to patients
          - id: review
            name: clinic.doctor
            duty: owns causal hypotheses and draft care protocols
            status: ready for case review
            copies: true
          - id: followup
            name: clinic.followup
            duty: owns follow-through on physician-approved plans
            status: tracking existing care plans
    annotation: human
scenes:
  - at: 0
    route: patient portal → clinic.intake
    text: "Case.014: “Help my doctor understand my persistent fatigue.”"
    active:
      - intake
    channel: portal
    states:
      intake: taking history for case.014
    copies: []
    gate: collecting context
    context: case.014 · intake
    copyLabel: 0 active copies · spawned as needed
  - at: 2000
    route: clinic.intake → clinic.lead
    text:
      History, records, medications, and lab results are assembled with their sources. No
      doctor copies are running yet.
    active:
      - intake
      - lead
    channel: records
    states:
      intake: assembling the case record
      lead: scoping case.014
    copies: []
    gate: collecting context
    context: case.014 · context assembled
    copyLabel: 0 active copies · spawned as needed
  - at: 4100
    route: clinic.lead → spawn doctor.01
    text:
      The case is ready. The lead spawns one doctor agent to review it; intake begins the
      next patient.
    active:
      - lead
      - review
      - intake
    channel: ""
    states:
      lead: starting the case review
      review: 1 copy reviewing case.014
      intake: starting case.015
    copies:
      - id: doctor.01
        status: reviewing the full case …
        active: true
    gate: initial review
    context: case.014 · review / case.015 · intake
    copyLabel: 1 copy spawned for case.014
  - at: 6700
    route: clinic.lead → spawn doctor.02
    text:
      The first review finds competing explanations. The lead adds a second copy for an
      independent assessment of the same case.
    active:
      - lead
      - review
    channel: ""
    states:
      lead: adding independent scrutiny
      review: 2 copies reviewing case.014
      intake: continuing case.015
    copies:
      - id: doctor.01
        status: examining possible causes
        active: true
      - id: doctor.02
        status: independently reviewing …
        active: true
    gate: review expanded
    context: case.014 · 1 → 2 doctor copies
    copyLabel: 2 copies spawned for case.014
  - at: 9300
    route: clinic.lead → spawn doctor.03
    text:
      The reviews disagree about the timeline. A third copy is spawned to challenge the
      evidence and identify what is missing.
    active:
      - lead
      - review
    channel: ""
    states:
      lead: responding to disagreement
      review: 3 copies reviewing case.014
      intake: continuing case.015
    copies:
      - id: doctor.01
        status: proposes an explanation
        active: true
      - id: doctor.02
        status: flags conflicting dates
        active: true
      - id: doctor.03
        status: checking the evidence …
        active: true
    gate: review expanded
    context: case.014 · 2 → 3 doctor copies
    copyLabel: 3 copies spawned for case.014
  - at: 11900
    route: doctor.01 ↔ doctor.02 ↔ doctor.03
    text:
      “Medication timing may explain this.” / “The symptoms may predate it.” / “Ask intake
      to verify the dates.”
    active:
      - review
    channel: ""
    states:
      lead: following the case conference
      review: challenging causal explanations
      intake: taking history for case.015
    copies:
      - id: doctor.01
        status: proposes an explanation
        active: true
      - id: doctor.02
        status: challenges the timeline
        active: true
      - id: doctor.03
        status: requests missing evidence
        active: true
    gate: disagreement under review
    context: case.014 · case conference
    copyLabel: 3 copies spawned for case.014
  - at: 14900
    route: clinic.doctor ↔ clinic.intake ↔ patient portal
    text:
      Intake confirms the earlier onset. The reviewers revise their analysis instead of
      voting the concern away.
    active:
      - intake
      - review
    channel: portal
    states:
      lead: tracking evidence changes
      review: revising the causal analysis
      intake: clarifying case.014 history
    copies:
      - id: doctor.01
        status: revises initial hypothesis
        active: true
      - id: doctor.02
        status: checks revised reasoning
        active: true
      - id: doctor.03
        status: preserves open questions
        active: true
    gate: analysis revised
    context: case.014 · clarification returned
    copyLabel: 3 copies spawned for case.014
  - at: 17800
    route: clinic.doctor × 3 → clinic.lead
    text:
      The lead combines evidence, alternative explanations, unresolved questions, and a
      draft care protocol.
    active:
      - lead
      - review
    channel: ""
    states:
      lead: preparing the physician brief
      review: checking the draft
      intake: continuing case.015
    copies:
      - id: doctor.01
        status: analysis + supporting sources
        active: true
      - id: doctor.02
        status: alternatives + objections
        active: true
      - id: doctor.03
        status: protocol + remaining gaps
        active: true
    gate: draft in preparation
    context: case.014 · synthesis
    copyLabel: 3 copies spawned for case.014
  - at: 20500
    route: clinic.lead → physician workspace
    text:
      The physician receives a reviewable draft, including the disagreement—not just a consensus
      answer.
    active:
      - human
      - lead
    channel: doctor
    states:
      human: review, revise, or approve
      lead: awaiting physician decision
      review: case review completed
      intake: continuing case.015
    copies:
      - id: doctor.01
        status: review complete
        active: false
      - id: doctor.02
        status: review complete
        active: false
      - id: doctor.03
        status: review complete
        active: false
    gate: awaiting physician approval
    context: case.014 · physician review
    ready: true
    copyLabel: 3 copies spawned for case.014
  - at: 23500
    route: clinic.lead → release case.014 doctor copies
    text:
      The temporary copies finish and are released; their analysis remains with the case.
      Intake continues while the physician reviews.
    active:
      - intake
      - human
    channel: portal
    states:
      human: case.014 awaiting review
      lead: retaining the case analysis
      review: ready to spawn when needed
      intake: preparing case.015
    copies: []
    gate: awaiting physician approval
    context: case.014 · review / case.015 · intake
    ready: true
    copyLabel: copies released · case analysis retained
order: 1
draft:
  title: case.014 / analysis.md
  note: Synthetic example · not a clinical recommendation
  sections:
    - title: Evidence correction
      text:
        The reported symptoms started before the medication change. The initial explanation
        was revised after checking the history. [intake timeline]
    - title: Causal analysis
      text:
        No single cause established. Alternative explanations, evidence gaps, and the reviewers’
        disagreement remain visible.
    - title: Draft care protocol
      text:
        Proposed assessment, treatment options, monitoring, and follow-up would be included
        for the physician to evaluate. Clinical details are omitted from this illustrative example.
    - title: Physician decision required
      text:
        Review the evidence; approve, revise, or request further work. Nothing has been
        released to the patient.
---

A clinical organization scales its case review as new questions arise. The human physician reviews the evidence and controls approval.
