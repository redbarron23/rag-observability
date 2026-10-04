# Resource Tagging Standard

Fictional sample document for the Contoso Cloud observability programme.

## Required tags

Every Azure and GCP resource must carry the tags `owner`, `cost-centre`, `environment`, `tier` and `service`. On GCP these are applied as labels.

## Allowed values

The `environment` tag must be one of `prod`, `staging`, `dev`. The `tier` tag must be `1`, `2` or `3` and drives coverage targets and alert routing.

## Enforcement

Azure Policy denies creation of resources missing required tags. On GCP an organisation policy and a nightly scan report untagged resources to the owning team. Untagged resources older than 30 days are escalated to the programme lead.
