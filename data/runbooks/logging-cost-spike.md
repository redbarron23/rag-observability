# Logging Cost Spike Runbook

Fictional sample document for the Contoso Cloud observability programme. Contoso Cloud, its teams, workspaces and projects are invented, and no customer data is used.

This runbook guides the investigation of a sudden rise in Azure Log Analytics or GCP Cloud Logging cost. Take amounts, rankings and dates from the billing data, never from this document. The steps below are recommendations: they do not establish a root cause, and no remediation is guaranteed to save money.

## Detection and triage

Confirm the spike in the billing data first: the cloud, the Azure subscription or GCP project, the day or days affected, and the amount, together with the billing period and currency the data covers. Note whether it is a one-day spike or a step change that persists.

Triage as high priority when the daily logging cost at least triples or the increase lasts more than two days. Check the change calendar and the onboarding waves in the cost smoothing plan before treating the spike as unexpected. Open a cost ticket that records the billing facts and the dataset they came from.

## Investigation: Azure Log Analytics

Find which tables drove the billable ingestion, day by day, in the affected workspace:

```
Usage
| where TimeGenerated > ago(30d) and IsBillable == true
| summarize BillableGB = sum(Quantity) / 1000 by DataType, bin(TimeGenerated, 1d)
| order by BillableGB desc
```

Then find the resources sending the most billable data to the top table, using `_BilledSize` grouped by `_ResourceId`. Check the `Operation` table for daily cap warnings, and compare the workspace's pricing tier and commitment tier with the ingested volume to see how much of the increase is overage. Review recent changes to diagnostic settings and data collection rules (DCRs).

## Investigation: GCP Cloud Logging

In Metrics Explorer, chart `logging.googleapis.com/billing/bytes_ingested` for the affected project, grouped by `resource_type` and `log`, to find which logs grew. The Logs Storage page shows the volume per log bucket.

In the billing export in BigQuery, filter `service.description = "Cloud Logging"` and group cost by `project.id`, `sku.description` and day. Check the admin activity audit log for recent changes to sinks and exclusions (`UpdateSink`, `UpdateExclusion`, `DeleteExclusion`) and to Data Access audit log settings.

## Likely causes (hypotheses)

These are hypotheses to test with the investigation data, not root causes. The timing of a spike alone does not prove its cause.

- A new service or onboarding wave started sending logs.
- Debug or verbose logging was left on after a release.
- A retry storm or error loop multiplied log volume.
- An exclusion filter, sampling rule or DCR transformation was removed or changed.
- A duplicated sink or diagnostic setting sends the same logs twice.
- Data Access audit logs, or a high-volume table such as container logs, were newly enabled.

## Remediation options

Every option below is a candidate, not a fix: verify savings in the billing data afterwards; savings are not guaranteed. Agree changes with the owning team, and never change retention, sampling or filters in production as part of the investigation itself.

- Retention: move high-volume Azure tables to the Basic Logs tier and shorten interactive retention; on GCP, shorten log bucket retention beyond the included 30 days. Retention mainly reduces storage cost, not ingestion cost.
- Sampling: sample debug and trace-level logs at the source or in the collector before ingestion.
- Exclusion filters: on GCP, add exclusion filters to the `_Default` sink for noisy logs; on Azure, drop rows or columns with a DCR transformation.
- Commitment tiers: see Commitment tiers below. A tier lowers the per-GB price; it does not reduce volume.

The Azure daily cap stops ingestion once reached and loses data; use it only as a temporary guard approved by the owning team.

## Verifying savings

Compare the daily cost for at least 14 days after a change with the same days before it, using the same billing data, period definition and currency. Report the measured difference, not the expected one. If the cost did not fall, roll the change back or try the next hypothesis. Savings are not guaranteed until the billing data shows them.

## Commitment tiers

Conflicting guidance: this section deliberately disagrees with the Commitment tiers section of programme/COST_SMOOTHING.md. That plan buys a 100 GB per day commitment tier once ingestion is stable for two consecutive months. This runbook requires three consecutive stable months after the last spike, and sizes the tier from the median daily ingestion of those months rather than a fixed 100 GB per day.

Do not buy or resize a commitment tier while a spike is under investigation. Until the programme reconciles the two documents, do not choose between them: escalate the commitment decision to FinOps.

## Escalation

Escalate to the team named in the resource's `owner` tag first, with the cost ticket. Escalate to FinOps for the cost centre when the spike continues for more than five days, when a commitment tier decision is needed, or when the owner is unknown. Escalate to the observability team when the cause involves shared workspaces, sinks or collection rules. A spike caused by an incident follows the incident process instead.
