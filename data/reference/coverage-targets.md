# Coverage Targets

Fictional sample document for the Contoso Cloud observability programme. All targets, windows and examples below are local sample policy, not universal standards or live cloud measurements.

## Coverage calculation

Report metrics, logs and traces coverage separately for each cloud and tier at a stated UTC snapshot time. Coverage = covered eligible units / all eligible units × 100%. Metrics and logs count resource instances; traces count services with declared critical journeys. Never average the three percentages into an overall coverage claim.

Freeze the resource inventory and service catalogue at the snapshot time. Record signal applicability for every unit. Known eligible units with missing or stale evidence remain in the denominator and do not count as covered. A zero eligible denominator is N/A, not 100%.

## Exclusions and unknown inventory

An exclusion needs a recorded reason, approver and expiry; for example, a component without request processing may have traces marked not applicable. Review exclusions at every snapshot. Unknown applicability is not an exclusion. Report unclassified units and missing inventory separately; percentages over the known subset are provisional and whole-estate coverage is unknown until inventory and applicability are reconciled. Incomplete inventory blocks readiness sign-off.

## Evidence and freshness

At snapshot time, metrics coverage requires every applicable golden signal to have a point in the last 15 minutes and points in at least 95% of its expected one-minute slots over the last 24 hours. Record the expected signals before measuring. Missing signals or stale points mean the resource is not covered.

Logs coverage requires a correlated test event from the resource within the last 24 hours, searchable centrally within 60 seconds of emission. Traces coverage requires a complete trace through every declared critical journey of the service within the last 24 hours. Controlled canaries are acceptable but must be marked synthetic; silence or sampling alone does not prove coverage.

Keep a per-unit evidence record: resource or service ID, cloud, tier, signal, applicability, snapshot time, observation window, query or test result, and event timestamps. Configuration alone does not prove delivery. These windows measure coverage; they do not establish all maturity criteria in the Observability Framework.

## Tier 1

Tier 1 resources are customer-facing production workloads. The production Tier 1 coverage target is 85% within 4 months and 100% within 9 months of programme start. Apply these fictional targets separately to each applicable signal and its own eligible denominator; trace targets refer to services rather than resource counts.

## Tier 2

Tier 2 resources are internal production workloads. The Tier 2 coverage target is 70% within 6 months and 95% within 12 months. Apply these fictional targets separately to each applicable signal and its own eligible denominator.

## Current status

In the fictional 2026-10-01 metrics-only snapshot, Azure coverage is 62% (62 / 100 eligible resources) against a Tier 1 interim target of 85%. GCP coverage is 0.47% (70 / 15,000, rounded) because onboarding only began last quarter; the GCP interim target is 40% by the end of next quarter and 100% for Tier 1 within 9 months. Logs and traces were not measured: their coverage is unknown, not zero. These figures cannot establish overall telemetry coverage or readiness.

## Worked coverage example

A fictional Azure Tier 1 snapshot has 12 inventoried resources and five catalogued services, with all applicability decisions reconciled. Metrics: 10 eligible resources, seven with fresh qualifying evidence, giving 70%. Logs: eight eligible resources, six covered, giving 75%. Traces: four eligible services, three covered, giving 75%. The other two metrics resources, four logs resources and one trace service have approved, unexpired exclusions.

A known eligible resource with stale evidence stays in its signal denominator and outside its numerator. If another resource is discovered with unknown applicability, report it as unclassified and the known-subset percentages as provisional. Do not silently exclude it or claim whole-estate readiness.
