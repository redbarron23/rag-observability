# Cobalt Checkout Service Standard

Fictional test document. All teams, service names, targets, and policies below are invented for testing a documentation assistant.

## Ownership

Cobalt Checkout is owned by the Atlas Commerce team. Its service identifier is `cobalt-checkout` and its production tier is Tier 1. Atlas Commerce maintains the dashboards, alert routing, and runbooks. The individual on-call rota is maintained separately and is not included in this document.

## Availability SLO

Cobalt Checkout has an availability SLO of 99.95% over a rolling 28-day window. Availability is the proportion of eligible checkout requests that return an HTTP status from 200 through 499. HTTP 500 through 599 responses are unsuccessful. Health probes and synthetic test requests are excluded from both the numerator and denominator.

## Error budget policy

The error budget permits unsuccessful responses for 0.05% of eligible requests. For example, one million eligible requests permit 500 unsuccessful responses. If the service consumes at least half of its 28-day error budget within seven days, Atlas Commerce pauses nonessential production deployments until a reliability review approves resuming them.

## Latency objective

The operational latency objective is a p95 checkout response time below 350 milliseconds, measured over five-minute windows. This latency objective is tracked separately from the availability SLO and does not change how unsuccessful requests are counted.

## Telemetry retention

Structured operational logs are retained for 30 days, debug logs for 7 days, and distributed traces for 14 days. Request logs include `service`, `environment`, `correlation_id`, and `trace_id`. These are fictional service-specific retention settings, not cloud-provider defaults.
