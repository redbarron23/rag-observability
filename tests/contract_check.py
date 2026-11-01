"""Validate API responses against the published contract in contract/v1/.

The published response schemas are open (consumers must ignore fields added in
minor versions). Provider tests validate against a closed copy instead, so a
field the server emits but the contract does not declare fails here rather
than shipping unannounced.
"""

import copy
import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "contract" / "v1"
SCHEMA_DIR = CONTRACT_DIR / "schemas"
EXAMPLES_DIR = CONTRACT_DIR / "examples"

REQUEST = "retrieve-request.schema.json"
RETRIEVE = "retrieve-response.schema.json"
HEALTH = "health-response.schema.json"
ERROR = "error-response.schema.json"


@cache
def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _closed(node):
    """Copy a schema with additionalProperties: false wherever properties are declared."""
    if isinstance(node, dict):
        # An "if" condition only tests some properties; closing it would never match.
        node = {key: value if key == "if" else _closed(value) for key, value in node.items()}
        if "properties" in node and "additionalProperties" not in node:
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [_closed(value) for value in node]
    return node


@cache
def validator(name: str, strict: bool = True) -> Draft202012Validator:
    schema = load_schema(name)
    return Draft202012Validator(_closed(copy.deepcopy(schema)) if strict else schema)


def response_schema(method: str, path: str, status: int) -> str:
    """The schema a response to this request must match."""
    if status != 200:
        return ERROR
    return {("POST", "/retrieve"): RETRIEVE, ("GET", "/health"): HEALTH}[(method, path)]


def errors(name: str, instance, strict: bool = True) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in validator(name, strict).iter_errors(instance)]


def assert_conforms(method: str, path: str, status: int, body, headers) -> None:
    """Assert a response matches the closed contract, body and version header."""
    name = response_schema(method, path.split("?", 1)[0], status)
    problems = errors(name, body)
    assert not problems, f"{method} {path} -> {status} breaks {name}: {problems}"
    if status != 200:
        assert body["error"]["code"] in load_schema(ERROR)["$defs"]["known_error_code"]["enum"]
    assert headers.get("X-RAG-Contract-Version") == body["contract_version"]
