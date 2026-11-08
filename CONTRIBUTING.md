# Contributing

Thanks for helping. This guide covers setting up, running the same checks as CI, changing the
eval set safely, opening a pull request and cutting a release. Work in progress is tracked on the
[open-source roadmap](https://github.com/users/redbarron23/projects/2).

Report security problems privately as described in [SECURITY.md](SECURITY.md), not in an issue.

## Development setup

Python 3.11 or newer. CI tests 3.11, 3.12 and 3.13.

```bash
git clone https://github.com/redbarron23/rag-observability.git
cd rag-observability
python -m venv .venv
source .venv/bin/activate
# Optional: CPU-only torch keeps the install small (CI does this)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` pulls in the core, Streamlit (`requirements-ui.txt`) and OpenTelemetry
(`requirements-otel.txt`) dependencies plus pytest, jsonschema and Ruff. The embedding model
(`all-MiniLM-L6-v2`) downloads on first use. No provider key is needed for anything below except
the `--with-llm` evals.

## Checks to run before a pull request

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs these steps in this order on
every pull request; run them locally first:

```bash
python -m ruff check .                 # lint
python -m pytest tests/ -q             # unit and contract tests
python ingest.py                       # index the bundled corpus into chroma_db/
python evals.py                        # retrieval evals, no API key
python adversarial_evals.py            # adversarial offline checks, separate fixture index
python telemetry_checks.py             # telemetry scenarios against a disposable Jaeger container
```

`telemetry_checks.py` needs Docker: it starts and removes a Jaeger container bound to loopback. It
makes no provider calls. If you can't run Docker, say so in the PR and let CI run it.

### Lint

Ruff is pinned to `>=0.17.3,<0.18` in `requirements-dev.txt`, because 0.x minor releases can add
rules. Configuration is in [`ruff.toml`](ruff.toml): lint only, no formatter, line-length errors
(`E501`) ignored. If a different Ruff is on your `PATH`, run the pinned range without installing
it:

```bash
uvx --from 'ruff>=0.17.3,<0.18' ruff check .
```

### Tests

`python -m pytest tests/ -q` needs no index and no provider key. `tests/test_contract.py` replays
every example in [`contract/v1/`](contract/v1/) against `api.py`. If you change the HTTP API
intentionally, regenerate the examples and review the diff:

```bash
UPDATE_CONTRACT_EXAMPLES=1 python -m pytest tests/test_contract.py
```

Any wire-format change also needs a contract version bump (`CONTRACT_VERSION` in `api.py` and
`x-contract-version` in each schema). The bump rules are in
[docs/http-api.md](docs/http-api.md#version-numbers).

### Ingest and retrieval evals

`python ingest.py` re-embeds only changed files; use `python ingest.py --rebuild` if the index
looks wrong. `python evals.py` then checks, for every answerable case, that the expected sources
and supporting sections are retrieved and that the expected facts are in the retrieved text. It
exits nonzero on failure. Add `--retrieval hybrid` to check the hybrid mode too, and `--verbose`
to see the retrieved chunks. Held-out questions are reported on their own lines and do not affect
the exit code.

### Paid generation evals (`--with-llm`)

```bash
export ANTHROPIC_API_KEY=...
python evals.py --with-llm --verbose
python evals.py --with-llm --retrieval hybrid --output run.json   # with a run artifact
```

These call the configured provider: 46 calls per retrieval mode (38 main cases plus 8 held-out).
A full run costs **about $0.42 per mode** with Claude Sonnet 5.5, measured on 8 November 2026 (see
the [benchmark](docs/benchmark.md)), so about $0.84 for both modes. Cost grows with the number of
cases; the run artifact records the actual token usage. `python adversarial_evals.py --with-llm`
adds 4 calls.

Run them when your change can affect answers: system or user prompts, citation handling, provider
or model code in `query.py`, the answer checks in `evals.py`, retrieval defaults (`k`, the
cutoff, the margin), chunking, or the corpus. Docs-only, CI-only and pure retrieval-plumbing
changes don't need them. Pull requests never run them in CI, so forks cannot reach the API key; a
maintainer can trigger the [LLM evals workflow](docs/guide.md#scheduled-llm-evals) on your branch
after review, and it also runs every Monday. Say in the PR whether you ran them, in which modes
and with which result.

## Adding eval cases

Cases live in [`evals.py`](evals.py):

| List | Purpose | Affects exit code |
| --- | --- | --- |
| `EVALS` | Answerable questions with `id`, `question`, `expected_sources`, `expected_sections` (exact `(source, section)` pairs), `expected_facts` and optionally `expected_relations` | Yes |
| `UNANSWERABLE_EVALS` | Questions the corpus can't answer; the model must decline | Yes |
| `HELDOUT_EVALS`, `HELDOUT_UNANSWERABLE_EVALS` | Generalisation check, never used for tuning | No |

To add a case:

1. Write the question and its expected sections and facts **from the source document**, before
   running retrieval. Label the minimum evidence a correct answer needs, not every useful section.
2. Add it to `EVALS` (or `UNANSWERABLE_EVALS`) and run `python evals.py` in both modes.
3. If it fails, fix the corpus, chunking or retrieval for a reason you can state, or record it as
   a known failure. Don't rewrite the expectation to match what came back.
4. Update the case counts in the README and `docs/guide.md`, and the eval-case table in
   [the guide](docs/guide.md#the-eval-cases).

Keep the held-out set held out:

- Don't look at held-out results while tuning the cutoff, margin, `k`, chunking or prompts, and
  don't change those settings to make a held-out case pass. A held-out failure is a finding to
  record in the PR, not a target.
- Don't move a question into the held-out lists after you have seen its distances or answers;
  that makes it a tuning case. New held-out questions must be written before any retrieval run
  against them, about sections no other case targets, and the PR should say so.
- Don't edit an existing held-out question or its expectations to make it pass. If one is wrong,
  fix it in a separate PR that explains the error.

Sections cited by obs-agent (the `obs-agent-*` cases) must keep their headings; see
[obs-agent integration cases](docs/guide.md#obs-agent-integration-cases).

## Pull requests

- Open or link an issue, and add it to the [roadmap](https://github.com/users/redbarron23/projects/2)
  if it isn't there. Use `Closes #N` in the description when the PR finishes the issue.
- Keep one topic per PR. Related issues that edit the same files can share one PR.
- CI must pass. Describe what you ran locally, including any `--with-llm` runs and their cost.
- Update the docs your change affects (README, `docs/`) and add a line under **Unreleased** in
  [CHANGELOG.md](CHANGELOG.md) for anything a user or API consumer would notice.
- PRs are **squash merged**, so the PR title becomes the commit message on `main`: write it as a
  short imperative sentence (for example "Add hybrid retrieval to the HTTP API").
- Only commit fictional or public documents to `data/` and `fixtures/`.

## Releasing

The project uses [Semantic Versioning](https://semver.org/) with Git tags `vX.Y.Z`. The project
version and the HTTP contract version are separate; how they relate is set out at the top of
[CHANGELOG.md](CHANGELOG.md#versioning).

Choosing the number (while the project is `0.x`):

- **PATCH** (`0.1.1`): bug fixes and documentation only.
- **MINOR** (`0.2.0`): new features, any contract MINOR or MAJOR bump, or a breaking change to CLI
  flags, defaults, environment variables, the index layout or the eval artifact schema.
- `1.0.0` will be cut once the CLI and index layout are considered stable; after that, breaking
  changes need a MAJOR bump.

Steps (maintainer):

1. In a release PR, rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`, add a
   fresh empty `## [Unreleased]` above it, state the contract version the release serves, and
   update the comparison links at the bottom. Squash merge it.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull --ff-only
   git tag -a vX.Y.Z -m "rag-observability vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. Publish the GitHub release with that version's changelog section as notes:

   ```bash
   awk '/^## \[X\.Y\.Z\]/{f=1; next} /^## \[|^\[.*\]: /{f=0} f' CHANGELOG.md > release-notes.md
   gh release create vX.Y.Z --verify-tag --title "vX.Y.Z" --notes-file release-notes.md
   rm release-notes.md
   ```

4. If the contract changed, tell consumers (obs-agent) which tag to re-vendor `contract/v1/` from.
