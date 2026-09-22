---
id: engineering
label: Engineering
filename: engineering.example
note: illustrative activity
duration: 19000
channels:
  - id: email
    label: email ↔ support
  - id: api
    label: api ↔ operations
  - id: slack
    label: slack ↔ director
roles:
  - id: director
    name: org.director
    duty: owns objectives, priorities, and escalations
    status: setting priorities
    children:
      - id: engineering
        name: org.engineering
        duty: owns product delivery and code quality
        status: maintaining the product
        copies: true
      - id: operations
        name: org.operations
        duty: owns availability and incident response
        status: monitoring production
      - id: support
        name: org.support
        duty: owns customer issues and follow-through
        status: watching the inbox
scenes:
  - at: 0
    route: 02:13 · scheduled checks → org.operations
    text: Night shift. Monitoring and inboxes remain active.
    active: []
    channel: ""
    states: {}
    copies:
      - id: copy.01
        status: available
        active: false
      - id: copy.02
        status: available
        active: false
      - id: copy.03
        status: available
        active: false
    context: Monday · 02:13
    copyLabel: engineer · three copies · one incident
  - at: 1800
    route: 02:14 · customer email → org.support
    text: “Checkout is failing.” Support contacts operations directly.
    active:
      - support
      - operations
    channel: email
    states:
      support: triaging customer report
      operations: checking error rates
    copies:
      - id: copy.01
        status: available
        active: false
      - id: copy.02
        status: available
        active: false
      - id: copy.03
        status: available
        active: false
    context: Monday · 02:14
    copyLabel: engineer · three copies · one incident
  - at: 3900
    route: 02:15 · org.operations → org.director → org.engineering
    text: Monitoring confirms the incident. The director makes it the top priority.
    active:
      - operations
      - director
      - engineering
    channel: api
    states:
      support: updating the customer
      operations: containing the incident
      director: "prioritizing incident #42"
      engineering: "assigning incident #42"
    copies:
      - id: copy.01
        status: "assigned #42"
        active: true
      - id: copy.02
        status: "assigned #42"
        active: true
      - id: copy.03
        status: "assigned #42"
        active: true
    context: Monday · 02:15
    copyLabel: engineer · three copies · one incident
  - at: 6300
    route: 02:17 · org.engineering → engineer × 3
    text: Three copies independently investigate the same incident.
    active:
      - engineering
      - operations
      - support
    channel: ""
    states:
      director: tracking the escalation
      engineering: comparing investigations
      operations: containing the incident
      support: updating the customer
    copies:
      - id: copy.01
        status: "investigating #42 …"
        active: true
      - id: copy.02
        status: "investigating #42 …"
        active: true
      - id: copy.03
        status: "investigating #42 …"
        active: true
    context: Monday · 02:17
    copyLabel: engineer · three copies · one incident
  - at: 9000
    route: 02:23 · engineer × 3 → org.engineering ↔ org.operations
    text: Engineering compares findings. Operations checks the proposed fix.
    active:
      - engineering
      - operations
    channel: ""
    states:
      director: tracking the escalation
      engineering: selecting the fix
      operations: verifying recovery
      support: awaiting recovery confirmation
    copies:
      - id: copy.01
        status: ✓ reproduced the failure
        active: true
      - id: copy.02
        status: ✓ isolated the regression
        active: true
      - id: copy.03
        status: ✓ tested a candidate fix
        active: true
    context: Monday · 02:23
    copyLabel: engineer · three copies · one incident
  - at: 11700
    route: 02:31 · org.operations → org.support → customer email
    text: Service is healthy. Support closes the loop with the customer.
    active:
      - operations
      - support
    channel: email
    states:
      director: reviewing incident outcome
      engineering: adding regression coverage
      operations: monitoring recovery
      support: confirming resolution
    copies:
      - id: copy.01
        status: "#42 complete"
        active: false
      - id: copy.02
        status: "#42 complete"
        active: false
      - id: copy.03
        status: "#42 complete"
        active: false
    context: Monday · 02:31
    copyLabel: engineer · three copies · one incident
  - at: 14500
    route: 06:00 · slack → org.director · next objective
    text: A new priority arrives. The same organization keeps working.
    active:
      - director
    channel: slack
    states:
      director: evaluating the next priority
      engineering: maintaining the product
      operations: monitoring production
      support: watching the inbox
    copies:
      - id: copy.01
        status: available
        active: false
      - id: copy.02
        status: available
        active: false
      - id: copy.03
        status: available
        active: false
    context: Monday · 06:00
    copyLabel: engineer · three copies · one incident
order: 0
---

A standing engineering organization handles an incident, coordinates parallel investigations, and continues to the next objective.
