---
policy_id: production-log-retention
version: platform-2026
effective_from: 2026-02-01
---
# Production Log Retention Policy

Fictional document: Northwind Systems, its teams and services are invented.

Owner: Platform Engineering. Status: in force. Effective from 1 February 2026.

## Retention target

Production application logs are retained for 90 days in the central logging platform. After 90 days they are deleted, and no archive copy is kept. Shorter retention keeps storage cost predictable.

## Scope

This policy applies to application logs from every production service. Audit logs and infrastructure metrics are out of scope.
