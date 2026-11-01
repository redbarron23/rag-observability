# Resource Tagging Standard

Fictional sample document for the Contoso Cloud observability programme. Contoso Cloud, its teams, cost centres and services are invented, and no customer data is used. The team and environment vocabulary matches the obs-agent synthetic demo so that both projects can be used together.

## Required tags

Every Azure and GCP resource must carry the tags `owner`, `cost-centre`, `environment`, `tier` and `service`. On GCP these are applied as labels.

Each Azure subscription and GCP project must also carry `owner`, `cost-centre` and `environment` itself (see Tag levels).

## Tag levels

Subscription and project level: each Azure subscription and GCP project carries `owner`, `cost-centre` and `environment`, describing the accountable team, the cost centre and the environment of the subscription or project as a whole.

Resource level: every resource inside a subscription or project carries all five required tags. Azure tags and GCP labels are not inherited from the subscription or project, so each resource must be tagged even when its values repeat those of its subscription or project.

## Owner

The `owner` tag names the accountable owning team, not a person. Its value must be one of the canonical team names in Team vocabulary. Individual engineers and resource creators are not valid `owner` values; they are recorded in the service catalogue.

This standard lists the allowed owner values only. It does not record which team owns any particular subscription, project or resource.

## Allowed values

The `environment` tag must be one of `prod`, `staging`, `dev`, `sandbox`. The `tier` tag must be `1`, `2` or `3` and drives coverage targets and alert routing.

Tier 1 is customer-facing production, Tier 2 is internal production and Tier 3 is everything else, including all staging, dev and sandbox resources. The `owner` tag must be a canonical team name, `cost-centre` must be that team's cost centre, and `service` must be an identifier registered in the service catalogue.

## Team vocabulary

| Canonical team (`owner`) | Cost centre (`cost-centre`) | Accepted legacy spellings |
|---|---|---|
| `payments` | `CC-1001` | `payment` |
| `data-platform` | `CC-2002` | `dataplatform`, `dp` |
| `ml` | `CC-2003` | `machine-learning` |
| `web` | `CC-3001` | `digital-web` |
| `identity` | `CC-4001` | `iam` |
| `observability` | `CC-4002` | `o11y`, `obs` |
| `security` | `CC-4003` | `secops`, `sec` |
| `finance-apps` | `CC-5001` | `finance`, `fin-apps` |

Legacy spellings are read as the canonical team when reviewing existing tags, but new tags must use the canonical name. Matching ignores case, treats spaces and underscores as hyphens, and ignores a `team-` prefix or `-team` suffix, so `Data Platform`, `data_platform` and `team-payments` all match a canonical team.

## Environment vocabulary

| Canonical environment (`environment`) | Accepted legacy spellings | Name tokens |
|---|---|---|
| `prod` | `production`, `prd` | `prod`, `prd`, `p` |
| `staging` | `stg`, `uat` | `stg`, `staging`, `uat` |
| `dev` | `development` | `dev`, `d`, `test` |
| `sandbox` | `sbx`, `poc` | `sbx`, `sandbox`, `poc` |

Legacy spellings are read as the canonical environment when reviewing existing tags, but new tags must use the canonical value. Name tokens are parts of a subscription or project name, such as `prd` in `pay-api-prd-weu`, that indicate its environment. A name token is evidence for an environment value; it does not replace the `environment` tag.

## Cost centre

Each team has exactly one cost centre, listed in Team vocabulary, and the `cost-centre` tag must be the cost centre of the `owner` team. Values are `CC-` followed by four digits. GCP label values must be lowercase, so GCP labels use `cc-1001` where Azure tags use `CC-1001`.

A subscription, project or resource whose cost centre disagrees with its owner team is non-compliant. The disagreement is resolved with Finance, not by choosing the most common value.

## Field mapping for obs-agent

obs-agent reviews labels on Azure subscriptions and GCP projects under its own field names. The tag keys of this standard map to them as follows.

| Policy tag | obs-agent field | How obs-agent treats it |
|---|---|---|
| `owner` | `team` | Inferred and reviewed |
| `environment` | `env` | Inferred and reviewed |
| `cost-centre` | `cost_center` | Derived from the selected team when all observed values for that team agree; never inferred on its own |
| `tier` | none | Not evaluated |
| `service` | none | Not evaluated |

## Requirements obs-agent does not evaluate

obs-agent infers `team` and `env` for Azure subscriptions and GCP projects and derives `cost_center` from the team. These requirements of this standard are outside its scope and must be checked by other means:

- Other tag keys: the `tier` and `service` tags, and any key other than `owner`, `environment` and `cost-centre`.
- Resource-level tags: only subscription and project labels are reviewed, not the tags on resources inside them.
- Tag key spelling: obs-agent reads values by its own field names and does not check that keys are spelled `owner`, `environment` and `cost-centre`.
- Enforcement: Azure Policy denials, the GCP organisation policy, the nightly scan and the 30-day escalation. obs-agent never writes tags or blocks resources; it produces a reviewed change plan.
- Ownership of a specific resource: the allowed values in this standard cannot establish which team owns a resource or raise the confidence of a suggestion.
- Telemetry coverage: whether resources send metrics, logs and traces (see Label completeness and coverage).

## Label completeness and coverage

Label completeness is the share of subscriptions, projects or resources that carry valid values for the required tags. Coverage is the share of cloud resources that send metrics, logs and traces to the central observability platform, as defined in the coverage targets.

The two are measured and reported separately, and neither substitutes for the other: a fully tagged resource may send no telemetry, and a resource with full coverage may be missing its `owner` tag.

## Enforcement

Azure Policy denies creation of resources missing required tags. On GCP an organisation policy and a nightly scan report untagged resources to the owning team. Untagged resources older than 30 days are escalated to the programme lead.
