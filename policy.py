"""Policy identity, effective dates and explicit supersession (issue #78).

A Markdown document may start with front matter naming the policy it states:

    ---
    policy_id: orders-queue-backlog-runbook
    version: 2
    effective_from: 2026-04-10
    effective_until: 2027-04-09
    supersedes: [orders-queue-backlog-runbook@1]
    ---

Every field is optional, but `version`, the dates and `supersedes` need a
`policy_id`. Documents without front matter are unversioned and stay usable.

Ingestion parses and validates the front matter (`parse_front_matter`,
`build_registry`) and stores the registry in the index manifest. Whether a
policy applies is decided at query time (`applicability`), because an
unchanged file can describe a policy that has since expired: source freshness
says whether the index matches the files, not whether their content is in
force.

Only an explicit `supersedes` reference from a policy in effect retires
another. A later `effective_from` alone never does: two policies in effect
with the same `policy_id` and no supersession between them are a conflict.
"""

from __future__ import annotations

import re
from datetime import date

FIELDS = ("policy_id", "version", "effective_from", "effective_until", "supersedes")
POLICY_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Statuses, in the order a document is classified.
INVALID = "invalid"          # its own front matter is malformed
UNVERSIONED = "unversioned"  # no policy_id (including documents without front matter)
FUTURE = "future"            # effective_from is after the as-of date
EXPIRED = "expired"          # effective_until is before the as-of date
SUPERSEDED = "superseded"    # a policy in effect explicitly supersedes it
CONFLICT = "conflict"        # another policy in effect has the same id, or supersession is circular
ACTIVE = "active"
UNAVAILABLE = "unavailable"  # the index has no policy registry (built before this feature)


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


def parse_front_matter(text: str) -> tuple[dict, str, list[str]]:
    """Split a document into (fields, body, errors).

    Front matter is a block that starts on the first line with `---` and ends
    at the next `---` line, holding `key: value` lines; `supersedes` takes a
    `[a@1, b@2]` list or a single reference. Without front matter the fields
    are empty and the body is the text unchanged. An unterminated block is an
    error and the text is kept as the body, so nothing is silently dropped.
    """
    if not text.startswith("---\n"):
        return {}, text, []
    end = text.find("\n---\n", 3)
    if end == -1:
        if text.rstrip("\n").endswith("\n---"):
            end = text.rstrip("\n").rfind("\n---")
        else:
            return {}, text, ["front matter starts with '---' but has no closing '---' line"]
    block, body = text[4:end], text[end + 5:]
    fields: dict = {}
    errors: list[str] = []
    for number, line in enumerate(block.split("\n"), start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep:
            errors.append(f"front matter line {number} is not 'key: value'")
            continue
        if key not in FIELDS:
            errors.append(f"unknown front matter key {key!r} (expected one of {', '.join(FIELDS)})")
            continue
        if key in fields:
            errors.append(f"duplicate front matter key {key!r}")
            continue
        value = value.strip()
        if key == "supersedes":
            items = value[1:-1].split(",") if value.startswith("[") and value.endswith("]") else [value]
            fields[key] = [_scalar(item) for item in items if _scalar(item)]
        else:
            fields[key] = _scalar(value) or None
    return fields, body.lstrip("\n"), errors


def _parse_date(value: str | None, field: str, errors: list[str]) -> date | None:
    if value is None:
        return None
    if DATE_RE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    errors.append(f"{field} {value!r} is not a valid YYYY-MM-DD date")
    return None


def ref(policy_id: str, version: str | None) -> str:
    """A policy reference, "<policy_id>@<version>" (or just the id)."""
    return f"{policy_id}@{version}" if version else policy_id


def validate(fields: dict, errors: list[str]) -> dict:
    """A registry record for one document's fields, with errors appended."""
    errors = list(errors)
    policy_id = fields.get("policy_id")
    version = fields.get("version")
    if policy_id is not None and not POLICY_ID_RE.fullmatch(policy_id):
        errors.append(f"policy_id {policy_id!r} must be lowercase letters, digits, '.', '_' or '-'")
    if version is not None and not VERSION_RE.fullmatch(version):
        errors.append(f"version {version!r} must be letters, digits, '.', '_' or '-'")
    start = _parse_date(fields.get("effective_from"), "effective_from", errors)
    until = _parse_date(fields.get("effective_until"), "effective_until", errors)
    if start and until and until < start:
        errors.append(f"effective_until {until} is before effective_from {start}")
    supersedes = fields.get("supersedes") or []
    for item in supersedes:
        target_id, _, target_version = item.partition("@")
        if not (POLICY_ID_RE.fullmatch(target_id) and VERSION_RE.fullmatch(target_version)):
            errors.append(f"supersedes reference {item!r} must be '<policy_id>@<version>'")
    if policy_id is None and any(fields.get(f) for f in FIELDS[1:]):
        errors.append("version, dates and supersedes need a policy_id")
    return {
        "policy_id": policy_id,
        "version": version,
        "effective_from": fields.get("effective_from"),
        "effective_until": fields.get("effective_until"),
        "supersedes": supersedes,
        "errors": errors,
    }


def build_registry(records: dict[str, dict]) -> dict[str, dict]:
    """Add corpus-wide errors to per-document records keyed by source path:
    duplicate policy references and supersedes references to no indexed
    policy. The result is stored in the index manifest."""
    registry = {source: {**record, "errors": list(record["errors"])}
                for source, record in sorted(records.items())}
    by_ref: dict[str, list[str]] = {}
    for source, record in registry.items():
        if record["policy_id"] and not record["errors"]:
            by_ref.setdefault(ref(record["policy_id"], record["version"]), []).append(source)
    for policy_ref, sources in by_ref.items():
        if len(sources) > 1:
            for source in sources:
                others = ", ".join(s for s in sources if s != source)
                registry[source]["errors"].append(f"{policy_ref} is also declared by {others}")
    for record in registry.values():
        for item in record["supersedes"]:
            if item not in by_ref:
                record["errors"].append(f"supersedes {item}, which no indexed document declares")
            elif item == ref(record["policy_id"], record["version"]):
                record["errors"].append("a policy cannot supersede itself")
    return registry


def _blank(status: str, as_of: date, errors: list[str] | None = None) -> dict:
    return {"status": status, "as_of": as_of.isoformat(), "policy_id": None, "version": None,
            "effective_from": None, "effective_until": None, "supersedes": [],
            "superseded_by": [], "pending_superseded_by": [], "conflicts_with": [],
            "errors": errors or []}


def unavailable(as_of: date) -> dict:
    return _blank(UNAVAILABLE, as_of,
                  ["the index has no policy registry; run python ingest.py to rebuild it"])


def applicability(registry: dict[str, dict], as_of: date) -> dict[str, dict]:
    """Each document's policy status on as_of, keyed by source path.

    Documents with errors are `invalid` and take no part in supersession or
    conflicts. A supersedes reference retires its target only once the
    superseding policy is in effect; before that it is listed under
    pending_superseded_by. Policies still in effect with the same policy_id
    are reported as a conflict with each other, as is circular supersession.
    """
    result: dict[str, dict] = {}
    in_effect: dict[str, str] = {}  # ref -> source, for valid policies not future or expired
    for source, record in registry.items():
        status = {**_blank(ACTIVE, as_of), **{k: record[k] for k in (
            "policy_id", "version", "effective_from", "effective_until")},
            "supersedes": list(record["supersedes"]), "errors": list(record["errors"])}
        if record["errors"]:
            status["status"] = INVALID
        elif not record["policy_id"]:
            status["status"] = UNVERSIONED
        elif record["effective_from"] and date.fromisoformat(record["effective_from"]) > as_of:
            status["status"] = FUTURE
        elif record["effective_until"] and date.fromisoformat(record["effective_until"]) < as_of:
            status["status"] = EXPIRED
        else:
            in_effect[ref(record["policy_id"], record["version"])] = source
        result[source] = status

    refs_of = {source: ref(r["policy_id"], r["version"])
               for source, r in registry.items() if r["policy_id"] and not r["errors"]}
    source_of = {policy_ref: source for source, policy_ref in refs_of.items()}
    for source, record in registry.items():
        if source not in refs_of:
            continue
        # A superseder not yet in effect only announces the change. One that
        # has since expired still retired its target: nothing revives it.
        key = "pending_superseded_by" if result[source]["status"] == FUTURE else "superseded_by"
        for target in record["supersedes"]:
            target_source = source_of.get(target)
            if target_source is not None:
                result[target_source][key].append(refs_of[source])

    # Circular supersession between policies in effect cannot be resolved.
    for policy_ref, source in in_effect.items():
        circular = [s for s in result[source]["superseded_by"]
                    if policy_ref in result[source_of[s]]["superseded_by"]]
        if circular:
            result[source]["status"] = CONFLICT
            result[source]["conflicts_with"].extend(circular)
            result[source]["errors"].append(f"circular supersession with {', '.join(circular)}")
    for source in in_effect.values():
        if result[source]["status"] == ACTIVE and result[source]["superseded_by"]:
            result[source]["status"] = SUPERSEDED

    # Same policy_id, both still active: report, never pick by date.
    active_by_id: dict[str, list[str]] = {}
    for policy_ref, source in in_effect.items():
        if result[source]["status"] in (ACTIVE, CONFLICT):
            active_by_id.setdefault(registry[source]["policy_id"], []).append(policy_ref)
    for refs in active_by_id.values():
        if len(refs) > 1:
            for policy_ref in refs:
                status = result[in_effect[policy_ref]]
                status["status"] = CONFLICT
                status["conflicts_with"].extend(r for r in refs if r != policy_ref
                                                and r not in status["conflicts_with"])
    for status in result.values():
        for key in ("superseded_by", "pending_superseded_by", "conflicts_with"):
            status[key] = sorted(set(status[key]))
    return result


def note(status: dict | None) -> str:
    """A short label for a chunk whose policy is not simply in force, shown to
    the model next to the document title; empty for active, unversioned and
    unknown documents."""
    if not status or status["status"] in (ACTIVE, UNVERSIONED, UNAVAILABLE):
        return ""
    label = ref(status["policy_id"], status["version"]) if status["policy_id"] else "document"
    detail = {
        SUPERSEDED: lambda: f"superseded by {', '.join(status['superseded_by'])}",
        FUTURE: lambda: f"not yet in effect (effective from {status['effective_from']})",
        EXPIRED: lambda: f"expired on {status['effective_until']}",
        CONFLICT: lambda: f"conflicts with {', '.join(status['conflicts_with'])}",
        INVALID: lambda: "policy metadata invalid; applicability unknown",
    }[status["status"]]()
    return f" [policy {label}: {detail} as of {status['as_of']}]"
