"""Provider checks for the published retrieval contract (contract/v1/).

Each example in contract/v1/examples/ is produced by the real api.py handler
serving a tiny fictional corpus on loopback, indexed with a deterministic
keyword embedder (no model download, no provider key, no LLM). The test
replays every example's request, checks the live response against the closed
schema, and compares it with the committed example after replacing only
run-specific values (timestamps, collection name, embedding revision) and
rounding distances.

After an intentional contract change, regenerate the examples with

    UPDATE_CONTRACT_EXAMPLES=1 python -m pytest tests/test_contract.py

and review the diff (and the contract version) before committing.
"""

import copy
import http.client
import json
import os
import re
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from contract_check import (
    CONTRACT_DIR,
    ERROR,
    EXAMPLES_DIR,
    HEALTH,
    REQUEST,
    RETRIEVE,
    SCHEMA_DIR,
    errors,
    load_schema,
    response_schema,
)

import api
import ingest

UPDATE = os.environ.get("UPDATE_CONTRACT_EXAMPLES") == "1"
API_KEY = "example-key"  # fictional; only the unauthorized and with-api-key examples use it

# A fictional organisation's docs. "Investigation" is longer than one chunk, so
# ingestion splits it in two and section completion can append the second half.
CORPUS = {
    "reference/tagging-policy.md": (
        "# Example Org Tagging Policy\n\n"
        "## Required tags\n\n"
        "Every resource in the fictional Example Org estate carries the `owner`, "
        "`cost-centre` and `environment` tags.\n\n"
        "## Tag exceptions\n\n"
        "Sandbox subscriptions may omit the cost-centre tag for up to 30 days.\n"
    ),
    "runbooks/queue-backlog.md": (
        "# Queue Backlog Runbook\n\n"
        "## Investigation\n\n"
        "Start with consumer lag on the orders queue. Compare the publish rate with the "
        "acknowledge rate for the last hour, and note when the two lines diverged. A growing "
        "gap with a steady publish rate points at slow consumers rather than a traffic spike, "
        "so check consumer CPU and recent deployments before scaling anything.\n\n"
        "Then open the dead-letter queue. Group the parked messages by error type and compare "
        "the first failure time with the divergence you found above. Replaying parked "
        "messages is safe only after the cause is fixed; record the replay in the incident "
        "channel so the next responder knows it already happened.\n"
    ),
    # Policy front matter (contract 1.1): v2 supersedes v1, v3 is announced for
    # 2099 and will supersede v2. Far-off dates keep the statuses stable.
    "policies/log-retention-v1.md": (
        "---\npolicy_id: example-log-retention\nversion: 1\neffective_from: 2020-01-01\n---\n"
        "# Log Retention Policy v1\n\n## Retention period\n\n"
        "Example Org keeps application logs for 30 days under this retention policy.\n"
    ),
    "policies/log-retention-v2.md": (
        "---\npolicy_id: example-log-retention\nversion: 2\neffective_from: 2021-01-01\n"
        "supersedes: [example-log-retention@1]\n---\n"
        "# Log Retention Policy v2\n\n## Retention period\n\n"
        "Example Org keeps application logs for 90 days under this retention policy.\n"
    ),
    "policies/log-retention-v3.md": (
        "---\npolicy_id: example-log-retention\nversion: 3\neffective_from: 2099-01-01\n"
        "supersedes: [example-log-retention@2]\n---\n"
        "# Log Retention Policy v3\n\n## Retention period\n\n"
        "Example Org will keep application logs for 180 days under this retention policy.\n"
    ),
}

# Cosine geometry chosen so the examples show each hit kind: A is the "required
# tags" direction; E is 0.65 from A (beyond the 0.6 cutoff, inside the 0.1
# exact-term margin); B and the second half of the runbook section are 1 apart.
A, B, D = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
E = [0.35, float(np.sqrt(1 - 0.35 ** 2)), 0.0]
KEYWORD_VECTORS = (("dead-letter", [0.0, 1.0, 0.0]), ("sandbox", E),
                   ("consumer lag", B), ("required", A), ("retention", [0.0, -1.0, 0.0]))


def keyword_embedder():
    def encode(texts, **_):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(next((v for word, v in KEYWORD_VECTORS if word in lowered), D))
        return np.array(rows)
    return Mock(encode=Mock(side_effect=encode))


PIN = {api.CONTRACT_HEADER: "1.0"}
REQUIRED_TAGS = {"question": "Which tags are required, including cost-centre?", "k": 3}

# name: (description, setup, server options, method, path, request headers, body)
CASES = {
    "retrieve-match": (
        "Ranked hits. The second hit is beyond max_distance and kept only because it contains "
        "the exact term 'cost-centre' (exact_term_match: true).",
        "current", {}, "POST", "/retrieve", PIN, REQUIRED_TAGS),
    "retrieve-section-expanded": (
        "k is 1, but the ranked hit's section was split in two at ingestion, so the other "
        "half is appended with expanded_from set. Expanded hits can exceed k and max_distance.",
        "current", {}, "POST", "/retrieve", PIN, {"question": "How do I investigate consumer lag?",
                                                  "k": 1}),
    "retrieve-policy-status": (
        "Contract 1.1: each hit carries its document's policy applicability. v2 supersedes v1 "
        "explicitly; v3 is not in effect until 2099 and is listed as pending for v2. Superseded "
        "and future guidance is labelled, never filtered out.",
        "current", {}, "POST", "/retrieve", {api.CONTRACT_HEADER: "1.1"},
        {"question": "How long does the retention policy keep logs?", "k": 3}),
    "retrieve-no-match": (
        "No chunk is within the cutoff: a successful 200 with empty hits, distinct from every "
        "failure.",
        "current", {}, "POST", "/retrieve", PIN, {"question": "What is the office wifi password?"}),
    "retrieve-stale": (
        "A source changed after ingestion. Without --require-fresh the server still answers, "
        "and freshness says which sources changed.",
        "stale", {}, "POST", "/retrieve", PIN, REQUIRED_TAGS),
    "retrieve-with-api-key": (
        "The server was started with RAG_API_KEY; the client sends it as a bearer token.",
        "current", {"api_key": API_KEY}, "POST", "/retrieve",
        {**PIN, "Authorization": f"Bearer {API_KEY}"},
        {"question": "Which tags are required?", "k": 1, "max_distance": 0.5}),
    "health-ok": (
        "Index present and sources unchanged since ingestion.",
        "current", {}, "GET", "/health", PIN, None),
    "health-degraded": (
        "Index present but stale: still a 200, with status 'degraded'.",
        "stale", {}, "GET", "/health", PIN, None),
    "error-invalid-request": (
        "k is outside 1-20. Request validation failures are 400 invalid_request.",
        "current", {}, "POST", "/retrieve", PIN, {"question": "Which tags are required?",
                                                  "k": 50}),
    "error-unauthorized": (
        "The server requires an API key and none was sent.",
        "current", {"api_key": API_KEY}, "POST", "/retrieve", PIN, REQUIRED_TAGS),
    "error-unsupported-contract-version": (
        "The client pinned a contract major version this server does not provide. It fails "
        "explicitly instead of returning hits the client cannot read.",
        "current", {}, "POST", "/retrieve", {api.CONTRACT_HEADER: "2"}, REQUIRED_TAGS),
    "error-index-unavailable": (
        "No index has been built yet (run python ingest.py). The server keeps running and "
        "serves the index once it exists.",
        "empty", {}, "POST", "/retrieve", PIN, REQUIRED_TAGS),
    "error-index-stale": (
        "With --require-fresh, a stale index is a 503 that carries the freshness detail.",
        "stale", {"require_fresh": True}, "POST", "/retrieve", PIN, REQUIRED_TAGS),
}

SETUP_TEXT = {
    "current": "Fictional corpus ingested; sources unchanged.",
    "stale": "Fictional corpus ingested, then reference/tagging-policy.md edited.",
    "empty": "No index has been built.",
}
RESPONSE_HEADERS = ("Content-Type", api.CONTRACT_HEADER, "WWW-Authenticate")

# Run-specific values replaced in the committed examples (all still schema-valid).
INGESTED_AT = "2026-01-15T09:30:00+00:00"
CHECKED_AT = "2026-01-15T09:31:00+00:00"
COLLECTION = "observability-docs-00000000000000000000000000000000"
EMBEDDING_REVISION = "0000000000000000000000000000000000000000"


@pytest.fixture
def make_index(tmp_path, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(return_value=keyword_embedder()))

    def _make(setup: str) -> Path:
        persist = tmp_path / "index"
        if setup == "empty":
            return persist
        data = tmp_path / "data"
        for rel, text in CORPUS.items():
            (data / rel).parent.mkdir(parents=True, exist_ok=True)
            (data / rel).write_text(text, encoding="utf-8")
        ingest.ingest_documents(str(data), str(persist))
        if setup == "stale":
            policy = data / "reference" / "tagging-policy.md"
            policy.write_text(policy.read_text() + "\nThe `tier` tag is now required too.\n")
        return persist
    return _make


def run_case(persist: Path, options: dict, method: str, path: str, headers: dict, body):
    service = api.RetrievalService(str(persist), embedder_factory=keyword_embedder,
                                   require_fresh=options.get("require_fresh", False))
    server = api.make_server(service, "127.0.0.1", 0, options.get("api_key"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        payload = None if body is None else json.dumps(body).encode()
        sent = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers={**sent, **headers})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
    response_headers = {h: resp.getheader(h) for h in RESPONSE_HEADERS if resp.getheader(h)}
    request_headers = {**sent, **headers}
    return request_headers, resp.status, response_headers, data


AS_OF = "2026-11-01"  # policy as_of is the server's current date; fixed in examples


def normalize(body: dict) -> dict:
    body = copy.deepcopy(body)
    corpus = body.get("corpus") or {}
    if corpus.get("provenance") == "available":
        corpus.update(ingested_at=INGESTED_AT, embedding_revision=EMBEDDING_REVISION)
    if "collection" in corpus:
        assert re.fullmatch(r"observability-docs-[0-9a-f]{32}", corpus["collection"])
        corpus["collection"] = COLLECTION
    for holder in (body, body.get("error") or {}):
        if "freshness" in holder:
            holder["freshness"]["checked_at"] = CHECKED_AT
    for hit in body.get("hits", []):
        hit["distance"] = round(hit["distance"], 4)
        if hit.get("policy"):
            hit["policy"]["as_of"] = AS_OF
    return body


def example_path(name: str) -> Path:
    return EXAMPLES_DIR / f"{name}.json"


@pytest.mark.parametrize("name", sorted(CASES))
def test_handler_reproduces_published_example(name, make_index):
    description, setup, options, method, path, headers, body = CASES[name]
    request_headers, status, response_headers, live = run_case(
        make_index(setup), options, method, path, headers, body)
    schema = response_schema(method, path, status)

    # The live response, not just the committed copy, matches the closed schema.
    assert not errors(schema, live), errors(schema, live)
    assert response_headers[api.CONTRACT_HEADER] == api.CONTRACT_VERSION

    example = {
        "description": description,
        "setup": SETUP_TEXT[setup] + (" Server started with RAG_API_KEY=example-key."
                                      if options.get("api_key") else "")
                 + (" Server started with --require-fresh." if options.get("require_fresh")
                    else ""),
        "request": {"method": method, "path": path, "headers": request_headers,
                    **({"body": body} if body is not None else {})},
        "response": {"status": status, "schema": f"schemas/{schema}",
                     "headers": response_headers, "body": normalize(live)},
    }
    target = example_path(name)
    if UPDATE:
        target.write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    assert target.exists(), f"Missing {target}; run with UPDATE_CONTRACT_EXAMPLES=1"
    assert json.loads(target.read_text(encoding="utf-8")) == example


def published_examples() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(EXAMPLES_DIR.glob("*.json"))}


def test_every_published_example_is_generated_and_valid():
    examples = published_examples()
    assert set(examples) == set(CASES), "examples/ must hold exactly the generated cases"
    for name, example in examples.items():
        response = example["response"]
        schema_file = CONTRACT_DIR / response["schema"]
        assert schema_file.parent == SCHEMA_DIR and schema_file.exists()
        assert not errors(schema_file.name, response["body"]), name
        # The open (published) schema accepts what the closed one does.
        assert not errors(schema_file.name, response["body"], strict=False), name


def test_request_examples_agree_with_the_request_schema():
    """Every documented request body the handler accepted is schema-valid, and the
    rejected one is schema-invalid, so client-side validation predicts the server."""
    for name, example in published_examples().items():
        request, response = example["request"], example["response"]
        if request["path"] != "/retrieve":
            continue
        rejected = response["body"].get("error", {}).get("code") == "invalid_request"
        assert bool(errors(REQUEST, request["body"])) == rejected, name
    accepted = [e for e in published_examples().values() if e["response"]["status"] == 200
                and e["request"]["path"] == "/retrieve"]
    assert len(accepted) >= 4


@pytest.mark.parametrize("body", [
    {}, {"question": "   "}, {"question": 5}, {"question": "x" * 2001}, {"question": "a\x00b"},
    {"question": "q", "k": 0}, {"question": "q", "k": 21}, {"question": "q", "k": True},
    {"question": "q", "max_distance": -0.1}, {"question": "q", "max_distance": 2.5},
    {"question": "q", "max_distance": "0.5"}, {"question": "q", "max_distance": None},
    {"question": "q", "n_results": 3}, ["question"],
])
def test_request_schema_rejects_what_the_handler_rejects(body):
    assert errors(REQUEST, body)
    with pytest.raises(api.ApiError):
        api.validate_request(body)


@pytest.mark.parametrize("body", [
    {"question": "q"}, {"question": "q" * 2000, "k": 20, "max_distance": 2},
    {"question": "q", "k": 1, "max_distance": 0},
])
def test_request_schema_accepts_what_the_handler_accepts(body):
    assert not errors(REQUEST, body)
    api.validate_request(body)


def test_schemas_declare_the_server_contract_version():
    from jsonschema import Draft202012Validator
    shared = None
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = load_schema(path.name)
        Draft202012Validator.check_schema(schema)
        assert schema["x-contract-version"] == api.CONTRACT_VERSION, path.name
        if path.name != REQUEST:
            # Each schema is self-contained for consumers; the shared parts must not drift.
            defs = {k: schema["$defs"][k] for k in ("contract_version", "corpus", "freshness")}
            assert shared in (None, defs), path.name
            shared = defs
    major = api.CONTRACT_VERSION.split(".")[0]
    assert re.fullmatch(load_schema(RETRIEVE)["$defs"]["contract_version"]["pattern"],
                        api.CONTRACT_VERSION)
    assert load_schema(RETRIEVE)["$defs"]["contract_version"]["pattern"].startswith(f"^{major}")


def test_known_error_codes_match_the_handler():
    source = Path(api.__file__).read_text(encoding="utf-8")
    raised = set(re.findall(r'ApiError\(\s*\d{3},\s*"([a-z_]+)"', source))
    assert raised == set(load_schema(ERROR)["$defs"]["known_error_code"]["enum"])


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("contract_version"),
    lambda b: b.update(contract_version="2.0.0"),
    lambda b: b["hits"][0].update(source_path=b["hits"][0].pop("source")),  # renamed field
    lambda b: b["hits"][0].update(policy_version="v9"),  # undeclared field (closed schema)
    lambda b: b.update(hits={}),
    lambda b: b["freshness"].update(status="old"),
    lambda b: b.update(generated=True),
    lambda b: b.update(content_trust="trusted"),
])
def test_closed_schema_catches_incompatible_responses(mutate):
    body = copy.deepcopy(published_examples()["retrieve-match"]["response"]["body"])
    mutate(body)
    assert errors(RETRIEVE, body)


def test_open_schema_tolerates_additive_fields():
    """What a pinned consumer sees after an additive (minor) change."""
    body = copy.deepcopy(published_examples()["retrieve-match"]["response"]["body"])
    body["contract_version"] = "1.1.0"
    body["hits"][0]["policy_version"] = "2026-03"
    body["corpus"]["new_field"] = True
    assert not errors(RETRIEVE, body, strict=False)
    health = copy.deepcopy(published_examples()["health-ok"]["response"]["body"])
    health["extra"] = {}
    assert not errors(HEALTH, health, strict=False)


@pytest.mark.parametrize("requested, ok", [
    (None, True), ("1", True), ("1.0", True), ("1.0.7", True), (" 1 ", True),
    ("1.1", True), ("1.1.3", True), ("1.2", False), ("2", False), ("0", False), ("", False), ("v1", False), ("1.x", False),
    ("latest", False),
])
def test_contract_version_pinning(requested, ok):
    if ok:
        api.check_contract_version(requested)
    else:
        with pytest.raises(api.ApiError) as err:
            api.check_contract_version(requested)
        assert (err.value.status, err.value.code) == (400, "unsupported_contract_version")


def test_incompatible_pin_is_rejected_on_every_endpoint(make_index):
    persist = make_index("current")
    for method, path, body in (("GET", "/health", None),
                               ("POST", "/retrieve", REQUIRED_TAGS)):
        _, status, headers, data = run_case(persist, {}, method, path,
                                            {api.CONTRACT_HEADER: "1.99"}, body)
        assert status == 400 and data["error"]["code"] == "unsupported_contract_version"
        assert "hits" not in data and headers[api.CONTRACT_HEADER] == api.CONTRACT_VERSION
