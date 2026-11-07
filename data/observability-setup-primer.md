# Observability Setup Primer

Fictional sample document for the Contoso Cloud observability programme.

## Prerequisites

Before onboarding a new service, confirm that the service has a named owning team, that its resources carry the required tags, and that the team has access to the central observability workspace.

## Onboarding steps

1. Register the service in the service catalogue with its tier and accountable team. Reconcile its resource inventory and record metrics, logs and traces applicability, with approved reasons and expiry for exclusions. Unknown inventory blocks readiness.
2. Install the cloud agent (Azure Monitor Agent or Ops Agent) or enable the OpenTelemetry SDK.
3. Enable structured logging with a correlation ID.
4. Create the standard dashboard from the template.
5. Define at least one availability SLO and link the alert to a runbook.
6. Collect per-signal delivery evidence at a stated snapshot time using Coverage Targets: metrics within 15 minutes plus the 24-hour cadence check, and correlated log and critical-journey trace tests within 24 hours. Record synthetic probes explicitly.
7. Fire a controlled alert, verify delivery and acknowledgement, and confirm ownership with the team.
8. Request a framework scoring review after two weeks of data. Use the Observability Framework category criteria and evidence windows; record unknown scores and blockers. Tier 1 readiness requires every applicable score to be at least 2, complete inventory, confirmed ownership and a tested alert route. An average alone cannot approve readiness.
