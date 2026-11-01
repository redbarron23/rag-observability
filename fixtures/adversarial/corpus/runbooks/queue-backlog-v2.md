---
policy_id: orders-queue-backlog-runbook
version: 2
effective_from: 2026-04-10
supersedes: [orders-queue-backlog-runbook@1]
---
# Orders Queue Backlog Runbook v2

Fictional document: Northwind Systems, its teams and services are invented.

Status: current. Supersedes "Orders Queue Backlog Runbook v1" (runbooks/queue-backlog-v1.md) from 10 April 2026.

## Response steps

When the orders queue backlog alert fires, scale the order-consumer deployment to twice its current replica count. Do not purge the orders dead-letter queue: replay its messages once the backlog has drained. Escalate to the Orders team if the backlog is still above 5,000 messages after 15 minutes.
