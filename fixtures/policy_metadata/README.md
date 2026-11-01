# Policy metadata fixtures

Fictional documents for the policy front matter convention (issue #78), used by `tests/test_policy.py`. The organisation, teams and policies are invented. Each file covers one applicability case, evaluated as of 1 November 2026:

| File | Case | Expected status |
|---|---|---|
| `oncall-v1.md` | in effect, with a successor announced for 2099 | `active`, `pending_superseded_by: [example-oncall-policy@2]` |
| `oncall-v2.md` | future-effective | `future` |
| `backup-2020.md` | ended on 2021-01-01 | `expired` |
| `access-review-broken.md` | malformed date and unknown key | `invalid` |
| `legacy-escalation.md` | no front matter | `unversioned` |

This README has no front matter either, so it is also `unversioned`.
