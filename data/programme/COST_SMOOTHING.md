# Log Analytics Cost Smoothing Plan

Fictional sample document for the Contoso Cloud observability programme.

## Problem

Log Analytics ingestion costs spike when many services onboard in the same month, which makes budgeting difficult.

## Plan

Onboarding is staggered into three waves, one per month, so that ingestion grows by no more than 20% month over month. Noisy debug logs are sampled at 10% before ingestion, and high-volume tables move to the Basic Logs tier after 30 days.

## Commitment tiers

Once ingestion is stable for two consecutive months, the programme purchases a 100 GB per day commitment tier to lower the per-GB price.
