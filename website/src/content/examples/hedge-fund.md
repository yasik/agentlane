---
id: hedge-fund
label: Hedge Fund
filename: fund.example
note: synthetic research · paper execution
duration: 33000
channels:
  - id: research
    label: filings / expert calls
  - id: data
    label: approved alt-data
  - id: execution
    label: paper broker API
roles:
  - id: pm
    name: portfolio.manager
    duty: vetted ideas + analytics + limits + fills → sizing, orders, P&L
    status: managing the portfolio
    children:
      - id: director
        name: research.director
        duty: theses + coverage gaps → vetted ideas, assignments
        status: directing sector coverage
        children:
          - id: analysts
            name: sector.analysts
            duty: nowcasts + calls + models → theses, sizing recommendations
            status: four sectors under review
            copies: true
            coverage: true
          - id: associate
            name: research.associate
            duty: analyst questions + raw data → clean models, diligence notes
            status: maintaining company models
          - id: data
            name: data.science
            duty: licensed data + KPI asks + usage approval → nowcasts, signals
            status: refreshing approved feeds
      - id: trader
        name: execution.trader
        duty: PM orders + borrow availability → fills, slippage, flow color
        status: watching liquidity and borrow
  - id: risk
    name: risk.manager
    duty: live positions + factor models → limits, exposure, stress reports
    status: monitoring portfolio exposure
    annotation: independent mandate
    independent: true
scenes:
  - at: 0
    route: research.director → sector.analysts
    text: Four sectors are under continuous coverage. Each analyst owns a research agenda.
    active:
      - director
      - analysts
    channel: research
    states: {}
    gate: research only
    context: coverage continues across the fund
    copyLabel: additional analysts spawned as needed
    copies: []
    coverage:
      - id: software.01
        status: reviewing filings
        active: false
      - id: healthcare.01
        status: reviewing disclosures
        active: false
      - id: energy.01
        status: tracking supply
        active: false
      - id: consumer.01
        status: monitoring demand
        active: false
  - at: 2400
    route: approved alt-data → data.science → consumer.01
    text:
      A permitted data refresh flags a possible demand change at fictional company C17.
      The analyst asks what is driving it.
    active:
      - data
      - analysts
    channel: data
    states:
      data: checking signal assumptions
      analysts: investigating C17
    gate: signal under review
    context: C17 · potential research lead
    copyLabel: additional analysts spawned as needed
    copies: []
    coverage:
      - id: software.01
        status: updating revenue models
        active: false
      - id: healthcare.01
        status: mapping research questions
        active: false
      - id: energy.01
        status: checking producer filings
        active: false
      - id: consumer.01
        status: questioning the C17 signal
        active: true
  - at: 5100
    route: consumer.01 → research.associate ↔ data.science
    text:
      The associate reconciles source data and the company model. Data Science checks coverage,
      seasonality, and KPI definitions.
    active:
      - analysts
      - associate
      - data
    channel: data
    states:
      associate: reconciling the C17 model
      data: validating the KPI nowcast
    gate: diligence in progress
    context: C17 · models + evidence
    copyLabel: additional analysts spawned as needed
    copies: []
    coverage:
      - id: software.01
        status: preparing expert-call questions
        active: false
      - id: healthcare.01
        status: updating company models
        active: false
      - id: energy.01
        status: updating demand scenarios
        active: false
      - id: consumer.01
        status: commissioning diligence
        active: true
  - at: 7900
    route: research.director → spawn consumer.02
    text:
      The opportunity needs deeper coverage. The director spawns another analyst to independently
      review the C17 demand thesis.
    active:
      - director
      - analysts
    channel: research
    states:
      director: expanding C17 coverage
      analysts: five analysts researching
    gate: coverage expanded
    context: C17 · additional analyst spawned
    copyLabel: C17 · 1 additional analyst spawned
    copies:
      - id: consumer.02
        status: independent demand review …
        active: true
    coverage:
      - id: software.01
        status: checking thesis assumptions
        active: false
      - id: healthcare.01
        status: tracking research catalysts
        active: false
      - id: energy.01
        status: reviewing expert-call notes
        active: false
      - id: consumer.01
        status: building the base thesis
        active: true
  - at: 10700
    route: research.director → spawn consumer.03
    text:
      A promotion effect could explain the signal. A third consumer analyst is assigned
      to challenge the thesis and investigate downside.
    active:
      - director
      - analysts
      - associate
    channel: research
    states:
      director: assigning a skeptical review
      analysts: six analysts researching
      associate: checking promotion effects
    gate: thesis challenged
    context: C17 · six analysts across four sectors
    copyLabel: C17 · 2 additional analysts spawned
    copies:
      - id: consumer.02
        status: cross-checking the model
        active: true
      - id: consumer.03
        status: challenging the downside …
        active: true
    coverage:
      - id: software.01
        status: reviewing filings
        active: false
      - id: healthcare.01
        status: reviewing disclosures
        active: false
      - id: energy.01
        status: tracking supply
        active: false
      - id: consumer.01
        status: testing the demand thesis
        active: true
  - at: 13700
    route: consumer.01 ↔ consumer.02 ↔ consumer.03 → research.director
    text:
      Analysts compare the nowcast, model, and objections. The director returns weak claims
      for revision and vets a narrower thesis.
    active:
      - analysts
      - associate
      - data
      - director
    channel: ""
    states:
      director: vetting evidence and objections
      associate: delivering diligence notes
      data: delivering qualified nowcast
    gate: research vetted
    context: C17 · thesis + sizing recommendation
    copyLabel: C17 · 2 additional analysts spawned
    copies:
      - id: consumer.02
        status: submitting supporting evidence
        active: true
      - id: consumer.03
        status: preserving downside risks
        active: true
    coverage:
      - id: software.01
        status: updating revenue models
        active: false
      - id: healthcare.01
        status: mapping research questions
        active: false
      - id: energy.01
        status: checking producer filings
        active: false
      - id: consumer.01
        status: revising the recommendation
        active: true
  - at: 16800
    route: research.director → portfolio.manager ↔ risk.manager
    text:
      The PM combines the vetted idea with portfolio analytics and proposes a position.
      Risk finds that the size breaches the sector limit.
    active:
      - pm
      - risk
    channel: ""
    states:
      pm: proposing position size
      risk: holding the proposed order
      trader: waiting for a permitted order
      director: tracking other coverage gaps
    gate: order held · sector limit
    context: C17 · portfolio decision
    copyLabel: C17 · 2 additional analysts spawned
    copies:
      - id: consumer.02
        status: documenting the evidence
        active: false
      - id: consumer.03
        status: tracking downside conditions
        active: false
    coverage:
      - id: software.01
        status: preparing expert-call questions
        active: false
      - id: healthcare.01
        status: updating company models
        active: false
      - id: energy.01
        status: updating demand scenarios
        active: false
      - id: consumer.01
        status: monitoring thesis assumptions
        active: false
  - at: 19800
    route: risk.manager → portfolio.manager → risk.manager
    text:
      The PM reduces the proposed size. Risk rechecks exposure and stress scenarios before
      clearing the revised paper order.
    active:
      - pm
      - risk
    channel: ""
    states:
      pm: revising the position size
      risk: revised order within mandate
      trader: preparing execution
    gate: revised paper order permitted
    context: C17 · revised sizing
    copyLabel: C17 · 2 additional analysts spawned
    copies:
      - id: consumer.02
        status: review complete
        active: false
      - id: consumer.03
        status: review complete
        active: false
    coverage:
      - id: software.01
        status: checking thesis assumptions
        active: false
      - id: healthcare.01
        status: tracking research catalysts
        active: false
      - id: energy.01
        status: reviewing expert-call notes
        active: false
      - id: consumer.01
        status: monitoring thesis assumptions
        active: false
  - at: 22900
    route: portfolio.manager → execution.trader → paper broker API
    text:
      The trader checks liquidity and applicable borrow constraints, then executes only
      the permitted paper order.
    active:
      - pm
      - trader
      - risk
    channel: execution
    states:
      pm: tracking the order
      trader: simulating order execution
      risk: checking resulting exposure
    gate: paper execution
    context: C17 · research retained; extra copies released
    copyLabel: extra copies released · sector coverage continues
    copies: []
    coverage:
      - id: software.01
        status: reviewing filings
        active: false
      - id: healthcare.01
        status: reviewing disclosures
        active: false
      - id: energy.01
        status: tracking supply
        active: false
      - id: consumer.01
        status: tracking the research thesis
        active: false
  - at: 25800
    route: execution.trader → portfolio.manager + risk.manager
    text:
      Simulated fills and slippage return to the PM. Risk updates exposures; the PM tracks
      the position and P&L as new data arrives.
    active:
      - trader
      - pm
      - risk
    channel: execution
    states:
      pm: tracking fills and P&L
      trader: reporting fills and slippage
      risk: updating exposure and stress
    gate: paper position under monitoring
    context: C17 · feedback to the portfolio
    copyLabel: extra copies released · sector coverage continues
    copies: []
    coverage:
      - id: software.01
        status: updating revenue models
        active: false
      - id: healthcare.01
        status: mapping research questions
        active: false
      - id: energy.01
        status: checking producer filings
        active: false
      - id: consumer.01
        status: monitoring the held thesis
        active: false
  - at: 28700
    route: healthcare.01 → research.director · next research question
    text:
      A different sector raises a new question. Research, risk monitoring, and portfolio
      management continue while the director assigns the next investigation.
    active:
      - director
      - analysts
      - risk
    channel: research
    states:
      director: assigning the next investigation
      risk: monitoring updated positions
      pm: reviewing portfolio performance
    gate: coverage continues
    context: C17 monitored · next idea in research
    copyLabel: extra copies released · sector coverage continues
    copies: []
    coverage:
      - id: software.01
        status: preparing expert-call questions
        active: false
      - id: healthcare.01
        status: escalating a coverage gap
        active: true
      - id: energy.01
        status: updating demand scenarios
        active: false
      - id: consumer.01
        status: monitoring the held thesis
        active: false
order: 2
---

A research organization maintains sector coverage, adds analysts when needed, and routes proposed positions through an independent risk mandate before paper execution.
