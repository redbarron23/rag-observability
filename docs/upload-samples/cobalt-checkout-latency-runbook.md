# Cobalt Checkout Latency Runbook

Fictional test document. The actions and thresholds below describe an invented service and must not be applied to a real system.

## Alert trigger

The Cobalt Checkout latency alert opens when the five-minute p95 response time remains above 350 milliseconds for ten consecutive minutes. Start by checking whether the increase affects all checkout requests or only a particular route or region.

## Triage

Inspect distributed traces for the checkout, inventory, and payment path. Compare time spent in the inventory call with time spent in the payment call. Use `correlation_id` to find the matching structured logs. Then inspect database connection-pool usage, queued requests, and recent configuration deployments.

## Escalation

Page the Atlas Commerce primary on-call when p95 checkout latency remains above 700 milliseconds for 15 consecutive minutes, or when HTTP 5xx responses exceed 2% of eligible checkout requests over a five-minute window. Route the page through the service catalogue; this runbook does not name the person currently on call.

## Mitigation approval

For this fictional service, the incident commander must approve a rollback before it is performed. Record the suspected change, the expected effect, and the post-rollback latency measurement in the incident timeline. This document contains no production commands or connection credentials.

## Recovery check

After mitigation, confirm that p95 latency stays below 350 milliseconds and HTTP 5xx responses stay below 0.1% for ten consecutive minutes. Keep the incident open if only one of these conditions recovers. The recovery threshold is an incident check, not a replacement for the availability SLO.
