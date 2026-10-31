# Cobalt Checkout Incident Review — 8 October 2026

Fictional historical incident for testing a documentation assistant. This record describes one invented event, not the service's present operating state.

## Timeline

The Cobalt Checkout incident began on 8 October 2026 at 13:05 UTC and ended at 13:24 UTC, lasting 19 minutes. A configuration deployment preceded the incident. The incident commander approved a rollback at 13:17 UTC, and the service met its recovery checks at 13:24 UTC.

## Customer impact

Checkout p95 latency peaked at 920 milliseconds and HTTP 5xx responses peaked at 3.8% over a five-minute window. Some customers could not complete checkout during the incident. This review does not contain a count of affected customers or an estimate of lost revenue.

## Root cause

A configuration deployment reduced the inventory client's maximum connection-pool size from 80 to 20. Inventory requests queued behind the smaller pool, which increased checkout response times and produced timeouts. The payment service was not the root cause of this incident.

## Resolution

The approved rollback restored the inventory connection-pool limit to 80 during the incident. Queued requests drained and checkout recovered. This is the historical resolution; it does not establish the current production configuration.

## Follow-up actions

Atlas Commerce owns the follow-up work: add a load test exercising 60 concurrent checkout requests, alert on inventory connection-pool wait time, and require review of connection-pool configuration changes. Completion dates and current follow-up status are not recorded here.
