# Data-First Approach

Fictional sample document for the Contoso Cloud observability programme.

## Principle

Reporting jobs read from the central metrics store first and only call cloud provider APIs when the stored data is stale. This keeps API usage and throttling low and makes reports reproducible.

## Freshness check

Each dataset records a `last_updated` timestamp. If the data is less than 6 hours old, the job uses it as is. If it is older, the job refreshes only the missing time window instead of re-pulling the whole history.

## Benefits

Checking freshness first avoids unnecessary cloud API calls, reduces the risk of rate limiting on the Azure Resource Graph and GCP Monitoring APIs, and cuts job runtime from minutes to seconds.
