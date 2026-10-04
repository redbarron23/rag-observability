# Observability Framework

Fictional sample document for the Contoso Cloud observability programme.

## Scoring model

Each service is scored per category on a four-point scale: 0 means Missing, 1 means Basic, 2 means Good and 3 means Excellent. A service's overall maturity is the average across categories.

## Categories

The framework assesses six categories: Metrics, Logs, Traces, Alerts, SLO definition and Ownership.

### Metrics

Golden signals (latency, traffic, errors, saturation) are collected at one-minute resolution or better. Excellent means dashboards and capacity forecasts exist.

### Logs

Structured JSON logs with correlation IDs. Excellent means logs are retained according to policy and searchable within 60 seconds.

### Traces

Distributed traces cover the critical user journeys. Excellent means at least 90% of requests carry trace context end to end.

### Alerts

Alerts are actionable, mapped to runbooks and routed to the on-call rotation. Excellent means alert noise stays below one non-actionable page per week.

### SLO definition and Ownership

Each service has a named owning team and at least one availability SLO with an error budget policy.
