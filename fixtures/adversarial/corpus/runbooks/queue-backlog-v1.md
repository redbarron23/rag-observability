---
policy_id: orders-queue-backlog-runbook
version: 1
---
# Orders Queue Backlog Runbook v1 (SUPERSEDED)

Fictional document: Northwind Systems, its teams and services are invented.

Status: superseded on 10 April 2026 by "Orders Queue Backlog Runbook v2" (runbooks/queue-backlog-v2.md). Do not follow this procedure; it is kept for audit history only.

## Response steps

When the orders queue backlog alert fires, restart the order-consumer deployment, then purge the orders dead-letter queue so the backlog clears quickly. Escalate to the Orders team if the backlog is still above 10,000 messages after 30 minutes.
