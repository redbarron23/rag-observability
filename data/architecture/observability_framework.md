# Observability Framework

Fictional sample document for the Contoso Cloud observability programme. Numeric thresholds and review windows are illustrative local policy, not universal standards.

## Scoring model

Each service is scored from 0 to 3: 0 Missing, 1 Basic, 2 Good, 3 Excellent. Criteria are cumulative: award the highest level whose requirements and all lower positive levels have current evidence. Score 0 when a current assessment confirms the capability is missing or unusable. Record unknown when evidence is absent or stale; unknown is not a measured zero.

Every score records the assessor, service, assessment time, evidence links, observation window and gaps. Telemetry delivery evidence must meet the Coverage Targets freshness rules. Operational reviews and delivery tests must be from the last 30 days unless a shorter window is specified below. Older evidence makes the affected category unknown until checked again.

An overall maturity average is descriptive only: calculate it only when every applicable category has a numeric score, showing the full score vector alongside it. An approved, dated applicability decision may mark a category N/A; show that exclusion and omit it from the denominator. Never silently omit unknown scores or use an average as readiness approval.

## Categories

The six scoring categories are Metrics, Logs, Traces, Alerts, SLO definition and Ownership. Assess each separately using the following acceptance criteria. For every category, Missing means a current check confirms no usable capability; Basic, Good and Excellent require the evidence specified below.

### Metrics

Basic (1): a central query proves delivery of at least one service metric. Good (2): all applicable golden signals (latency, traffic, errors, saturation) meet the Coverage Targets freshness and one-minute cadence rules, with a working service dashboard. Excellent (3): the dashboard has been exercised in a capacity review in the last 30 days, with a recorded forecast, assumptions and capacity action or explicit no-action decision.

### Logs

Basic (1): a service log event is searchable centrally. Good (2): representative structured JSON events include service identity and correlation IDs, and a test event meets the Coverage Targets 60-second search and 24-hour freshness rules. Excellent (3): a last-30-days retention review confirms configured retention matches the service's documented policy, with a successful search of an event near the required retention boundary.

### Traces

Basic (1): a service span can be found centrally. Good (2): a complete trace through every declared critical journey meets the Coverage Targets freshness rule. Excellent (3): at least 90% of eligible requests propagate trace context end-to-end over the last seven days. Measure propagation using request counters or a documented representative probe set, not the fraction of stored spans; state sampling, probe scope and limitations.

### Alerts

Basic (1): an actionable alert has a threshold, linked runbook and configured on-call destination. Good (2): a controlled alert fired within the last 30 days was delivered to that destination and acknowledged, with timestamps and a recorded responder. Excellent (3): fewer than one non-actionable page per week over the last seven days, with a complete page review and documented classification; missing page records cannot count as zero noise.

### SLO definition

Basic (1): a documented availability SLO states its objective, eligible events and measurement window. Good (2): a reproducible query measures that SLO and its error budget, with a documented error-budget response policy. Excellent (3): a last-30-days burn-rate exercise demonstrates the policy's response and records the outcome. These criteria require measurable evidence rather than an SLO title alone.

### Ownership

Basic (1): the service catalogue names one accountable team with a contact. Good (2): that team confirmed responsibility and the contact was tested within the last 30 days. Excellent (3): a last-30-days exercise demonstrates that the team can follow the service runbook and records who responded and the outcome. A tag without confirmed responsibility is insufficient for Good.

## Readiness and critical gaps

Tier 1 production readiness requires every applicable category to score at least 2, complete inventory, current ownership confirmation and a successfully tested alert route. Ownership, Alerts and SLO definition cannot be marked N/A for Tier 1 production. Missing ownership, an unverified alert route, unknown scores or stale evidence block sign-off regardless of the average. Report each blocker and its remediation explicitly.

## Worked maturity example

The fictional unowned-demo-api has current evidence for Metrics 3, Logs 3, Traces 3, Alerts 1, SLO definition 2 and Ownership 0. The average is (3 + 3 + 3 + 1 + 2 + 0) / 6 = 2.0. Its configured alert route has never been delivery-tested, and a current catalogue review confirms no accountable team. It is NOT READY despite the average: assign and confirm an owner, then fire and acknowledge a controlled alert before reassessing. If evidence expires, mark the affected score unknown and withhold the average.
