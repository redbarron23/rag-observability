"""Policy identity, effective dates and supersession (issue #78). Offline."""

import json
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

import ingest
import policy
from query import RAGEngine, anthropic_content, deepseek_messages

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "policy_metadata"
AS_OF = date(2026, 11, 1)


def record(policy_id=None, version=None, effective_from=None, effective_until=None,
           supersedes=(), errors=()):
    return {"policy_id": policy_id, "version": version, "effective_from": effective_from,
            "effective_until": effective_until, "supersedes": list(supersedes),
            "errors": list(errors)}


def statuses(records: dict, as_of=AS_OF) -> dict:
    return {source: s["status"] for source, s in
            policy.applicability(policy.build_registry(records), as_of).items()}


# ── Front matter parsing and validation ────────────────────────────────

def test_document_without_front_matter_is_unchanged():
    text = "# Title\n\nBody text.\n"
    assert policy.parse_front_matter(text) == ({}, text, [])


def test_front_matter_is_parsed_and_stripped():
    text = ("---\npolicy_id: a-policy\nversion: \"2\"\neffective_from: 2026-01-01\n"
            "supersedes: [a-policy@1, 'b-policy@3']\n---\n\n# Title\n")
    fields, body, errors = policy.parse_front_matter(text)
    assert fields == {"policy_id": "a-policy", "version": "2", "effective_from": "2026-01-01",
                      "supersedes": ["a-policy@1", "b-policy@3"]}
    assert body == "# Title\n" and errors == []


@pytest.mark.parametrize("text, message", [
    ("---\npolicy_id: a\n# no closing line\n", "no closing '---'"),
    ("---\nowner: Security\n---\nbody", "unknown front matter key 'owner'"),
    ("---\npolicy_id: a\npolicy_id: b\n---\nbody", "duplicate front matter key"),
    ("---\njust words\n---\nbody", "is not 'key: value'"),
])
def test_front_matter_errors(text, message):
    _, body, errors = policy.parse_front_matter(text)
    assert any(message in e for e in errors)
    if "closing" in message:
        assert body == text  # nothing silently dropped


@pytest.mark.parametrize("fields, message", [
    ({"policy_id": "a", "effective_from": "2026-13-45"}, "not a valid YYYY-MM-DD date"),
    ({"policy_id": "a", "effective_from": "1 Jan 2026"}, "not a valid YYYY-MM-DD date"),
    ({"policy_id": "a", "effective_from": "2026-02-01", "effective_until": "2026-01-01"},
     "is before effective_from"),
    ({"policy_id": "Bad ID"}, "policy_id 'Bad ID'"),
    ({"policy_id": "a", "supersedes": ["a"]}, "must be '<policy_id>@<version>'"),
    ({"version": "2"}, "need a policy_id"),
])
def test_field_validation(fields, message):
    assert any(message in e for e in policy.validate(fields, [])["errors"])


def test_registry_flags_duplicates_dangling_and_self_references():
    registry = policy.build_registry({
        "a.md": record("p", "1"), "b.md": record("p", "1"),
        "c.md": record("q", "1", supersedes=["missing@1"]),
        "d.md": record("r", "1", supersedes=["r@1"]),
    })
    assert any("also declared by b.md" in e for e in registry["a.md"]["errors"])
    assert any("no indexed document declares" in e for e in registry["c.md"]["errors"])
    assert any("cannot supersede itself" in e for e in registry["d.md"]["errors"])


# ── Applicability ──────────────────────────────────────────────────────

def test_statuses_for_each_case():
    assert statuses({
        "legacy.md": record(),
        "active.md": record("p", "1", "2025-01-01"),
        "future.md": record("q", "1", "2099-01-01"),
        "expired.md": record("r", "1", "2020-01-01", "2021-01-01"),
        "invalid.md": record("s", "1", errors=["bad date"]),
    }) == {"legacy.md": "unversioned", "active.md": "active", "future.md": "future",
           "expired.md": "expired", "invalid.md": "invalid"}


def test_explicit_supersession_retires_the_target():
    result = policy.applicability(policy.build_registry({
        "v1.md": record("p", "1", "2025-01-01"),
        "v2.md": record("p", "2", "2026-01-01", supersedes=["p@1"]),
    }), AS_OF)
    assert result["v1.md"]["status"] == "superseded"
    assert result["v1.md"]["superseded_by"] == ["p@2"]
    assert result["v2.md"]["status"] == "active"


def test_a_newer_date_alone_does_not_establish_authority():
    result = policy.applicability(policy.build_registry({
        "platform.md": record("retention", "platform", "2026-02-01"),
        "security.md": record("retention", "security", "2026-03-01"),
    }), AS_OF)
    assert {s["status"] for s in result.values()} == {"conflict"}
    assert result["platform.md"]["conflicts_with"] == ["retention@security"]


def test_future_supersession_is_pending_and_takes_effect_on_its_date():
    records = {"v1.md": record("p", "1", "2025-01-01"),
               "v2.md": record("p", "2", "2027-01-01", supersedes=["p@1"])}
    before = policy.applicability(policy.build_registry(records), AS_OF)
    assert before["v1.md"]["status"] == "active"
    assert before["v1.md"]["pending_superseded_by"] == ["p@2"]
    after = policy.applicability(policy.build_registry(records), date(2027, 1, 1))
    assert after["v1.md"]["status"] == "superseded" and after["v2.md"]["status"] == "active"


def test_superseded_policy_does_not_revive_when_successor_expires():
    assert statuses({
        "v1.md": record("p", "1", "2024-01-01"),
        "v2.md": record("p", "2", "2025-01-01", "2026-01-01", supersedes=["p@1"]),
    })["v1.md"] == "superseded"


def test_circular_supersession_is_a_conflict():
    result = policy.applicability(policy.build_registry({
        "a.md": record("p", "a", supersedes=["p@b"]),
        "b.md": record("p", "b", supersedes=["p@a"]),
    }), AS_OF)
    assert {s["status"] for s in result.values()} == {"conflict"}
    assert any("circular" in e for e in result["a.md"]["errors"])


def test_invalid_documents_take_no_part_in_supersession():
    result = policy.applicability(policy.build_registry({
        "v1.md": record("p", "1"),
        "v2.md": record("p", "2", supersedes=["p@1"], errors=["bad date"]),
    }), AS_OF)
    assert result["v1.md"]["status"] == "active" and result["v2.md"]["status"] == "invalid"


def test_note_labels_only_statuses_that_need_attention():
    result = policy.applicability(policy.build_registry({
        "v1.md": record("p", "1"), "v2.md": record("p", "2", supersedes=["p@1"]),
        "legacy.md": record()}), AS_OF)
    assert policy.note(result["v2.md"]) == policy.note(result["legacy.md"]) == ""
    assert policy.note(result["v1.md"]) == " [policy p@1: superseded by p@2 as of 2026-11-01]"
    assert policy.note(None) == ""


# ── Through ingestion, retrieval and the prompt ────────────────────────

@pytest.fixture
def engine(tmp_path, monkeypatch, capsys):
    data, persist = tmp_path / "data", tmp_path / "index"
    shutil.copytree(FIXTURES, data)
    embedder = Mock()
    embedder.encode.side_effect = lambda texts, **_: Mock(
        tolist=Mock(return_value=[[1.0, 0.0]] * len(texts)))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Mock(return_value=embedder))
    ingest.ingest_documents(str(data), str(persist))
    output = capsys.readouterr().out
    return RAGEngine(str(persist), embedder=embedder), output, persist


def test_fixtures_through_ingestion_and_retrieval(engine):
    rag, output, _ = engine
    assert "access-review-broken.md: unknown front matter key 'owner'" in output
    assert "not a valid YYYY-MM-DD date" in output
    assert {s: v["status"] for s, v in rag.policy_statuses(AS_OF).items()} == {
        "README.md": "unversioned", "oncall-v1.md": "active", "oncall-v2.md": "future",
        "backup-2020.md": "expired", "access-review-broken.md": "invalid",
        "legacy-escalation.md": "unversioned"}
    chunks = rag.retrieve("on-call response times", n_results=10, max_distance=None, as_of=AS_OF)
    assert all(not c["text"].startswith("---") and "policy_id:" not in c["text"] for c in chunks)
    by_source = {c["source"]: c["policy"] for c in chunks}
    assert by_source["oncall-v1.md"]["pending_superseded_by"] == ["example-oncall-policy@2"]
    # Applicability labels, it does not filter: future and expired guidance is still returned.
    assert {"oncall-v2.md", "backup-2020.md"} <= by_source.keys()


def test_status_labels_reach_both_providers_prompts(engine):
    rag, _, _ = engine
    chunks = [c for c in rag.retrieve("x", n_results=10, max_distance=None, as_of=AS_OF)
              if c["source"] in ("oncall-v2.md", "oncall-v1.md")]
    titles = [block["title"] for block in anthropic_content("q", chunks)[:-1]]
    assert any("not yet in effect (effective from 2099-01-01)" in t for t in titles)
    assert not any("[policy" in t for t, c in zip(titles, chunks, strict=True)
                   if c["source"] == "oncall-v1.md")
    user = deepseek_messages("q", chunks)[-1]["content"]
    assert "[policy example-oncall-policy@2: not yet in effect" in user


def test_index_without_registry_reports_unavailable_and_reingest_adds_it(engine):
    rag, _, persist = engine
    collection = rag.collection
    manifest = json.loads(collection.metadata["corpus_manifest"])
    del manifest["policies"]
    collection.modify(metadata={"corpus_manifest": json.dumps(manifest)})
    old = RAGEngine(str(persist), embedder=rag.embedder)
    assert old.policy_statuses(AS_OF) is None
    chunk = old.retrieve("x", n_results=1, max_distance=None, as_of=AS_OF)[0]
    assert chunk["policy"]["status"] == "unavailable"
    # Same files, but the registry is missing, so ingest rebuilds (reusing embeddings).
    ingest.ingest_documents(str(persist.parent / "data"), str(persist))
    assert RAGEngine(str(persist), embedder=rag.embedder).policy_statuses(AS_OF) is not None


def test_source_freshness_is_independent_of_applicability(engine):
    _, _, persist = engine
    data = persist.parent / "data"
    assert ingest.check_freshness(str(persist), str(data))["status"] == "current"
    # The expired backup policy is still "current" as an indexed file.
    rag = RAGEngine(str(persist), embedder=Mock())
    assert rag.policy_statuses(AS_OF)["backup-2020.md"]["status"] == "expired"

