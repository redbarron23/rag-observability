# Retrieval benchmark: keyword vs vector vs hybrid

Issue [#34](https://github.com/redbarron23/rag-observability/issues/34) asks whether retrieval
complexity and hosted generation give a measurable benefit on this corpus before more retrieval
features are built ([#8](https://github.com/redbarron23/rag-observability/issues/8)). This page
records the automated part of that comparison, measured on 8 November 2026. The human part
(manual document search) has **not been measured yet**. See
[Human comparison](#human-comparison-not-yet-measured).

All figures below come from two committed files:

- [`benchmarks/retrieval-2026-11-08.json`](benchmarks/retrieval-2026-11-08.json):
  `python benchmark.py`, retrieval only, no provider calls.
- [`benchmarks/generation-vector-2026-11-08.json`](benchmarks/generation-vector-2026-11-08.json):
  `python evals.py --with-llm --output ...`, one generated-answer run in vector mode. No answer
  text is included.

## What was compared

| Retriever | What it does |
|---|---|
| `keyword` | BM25 top-k over the chunk text. It uses no embeddings, no distance cutoff and no split-section completion. It returns any chunk that shares a term with the question. This is the new baseline (`KeywordRetriever` in [`benchmark.py`](../benchmark.py)). It reuses the BM25 scorer that hybrid mode uses. |
| `vector` | The default pipeline: cosine top-k with MiniLM embeddings, the 0.6 distance cutoff and exact-term margin, and split-section completion. |
| `hybrid` | `--retrieval hybrid`: vector and BM25 rankings fused by RRF (`k = 60`), with the same cutoff and completion. |

All three read the same indexed chunks. They are scored with the same checks CI runs
(`evals.run_retrieval_evals`) at k = 6.

## Conditions

| | |
|---|---|
| Date | 2026-11-08 (UTC), commit `d097932` (clean tree) |
| Corpus | Bundled fictional corpus: 9 documents, 82 chunks, 4,492 words. Corpus SHA-256 `d3c05b06…3130f`, chunk size 500 |
| Embedding | `all-MiniLM-L6-v2` (revision `1110a243`), sentence-transformers 6.1.0, torch 2.14.1, chromadb 1.5.9, CPU |
| Settings | k = 6, max distance 0.6, exact-term margin 0.1, section completion up to 3 chunks, RRF k = 60, BM25 k1 = 1.5, b = 0.75 |
| Questions | Main set: 32 answerable (37 supporting sections) and 6 unanswerable. Held-out set: 6 answerable and 2 unanswerable |
| Machine | Apple M2, 8 cores, 16 GB, macOS 26.6.2, Python 3.11.5 |
| Warm timing | Each of the 38 main-set questions retrieved 5 times (190 queries) in one process, after a full quality pass had warmed it |
| Cold timing | Median of 3 fresh Python processes per mode. Imports, retriever construction and the first query are timed separately. The OS file cache was **not** flushed, so the model files came from memory. A first run after a reboot will be slower |

## Retrieval-only results

Main set, top-6:

| Metric | keyword | vector | hybrid |
|---|---|---|---|
| Answerable cases passing all checks | 30/32 | **32/32** | **32/32** |
| Document attribution | 32/32 | 32/32 | 32/32 |
| Fact evidence in retrieved text | 31/32 | 32/32 | 32/32 |
| Supporting-section recall @1 | 21/37 | 25/37 | **27/37** |
| Supporting-section recall @3 | 30/37 | **35/37** | 33/37 |
| Supporting-section recall @6 | 35/37 | **37/37** | **37/37** |
| MRR of first supporting section | 0.780 | 0.871 | 0.898 |
| Unanswerable questions rejected | 1/6 | **6/6** | **6/6** |

Held-out set (not used for tuning):

| Metric | keyword | vector | hybrid |
|---|---|---|---|
| Answerable passing | 6/6 | 6/6 | 6/6 |
| Section recall @1 / @3 / @6 | 5 / 6 / 6 of 6 | 5 / 6 / 6 of 6 | 5 / 6 / 6 of 6 |
| Unanswerable rejected | 0/2 | 1/2 | 1/2 |

Latency per query:

| | keyword | vector | hybrid |
|---|---|---|---|
| Warm median | 0.52 ms | 7.24 ms | 8.87 ms |
| Warm p95 | 0.57 ms | 10.54 ms | 11.04 ms |
| Cold: construction | 0.045 s | 2.69 s | 2.72 s |
| Cold: first query | 0.002 s | 0.135 s | 0.156 s |
| Cold: imports | 3.56 s | 3.26 s | 3.50 s |
| Cold: whole process | 4.18 s | 6.79 s | 7.01 s |

Vector and hybrid construction is mostly loading the embedding model. Keyword's import time is
inflated: `KeywordRetriever` imports its BM25 scorer from `query.py`, which also imports
sentence-transformers and chromadb. A standalone keyword search would not pay that cost.

### Where the modes differ

Of the 32 main-set cases, 17 rank at least one supporting section differently between modes.
The ones that matter:

- **Keyword misses two cases.** `tagging-standard` doesn't retrieve *Required tags*, and the
  fact `cost-centre` is missing from its retrieved text. `obs-agent-suspect-env` doesn't retrieve
  *Allowed values*. Vector finds both at rank 2.
- **Keyword ranks prose questions lower.** Examples are `cost-smoothing` (rank 4 against 1),
  `exact-tag-values` (4 against 1) and `coverage-denominator` (2 against 1).
- **Keyword ranks exact figures higher.** `exact-basic-logs` and `exact-gcp-coverage-figure`
  are at rank 1 in keyword and hybrid, against 3 and 2 in vector. Hybrid keeps those gains.
- **Hybrid against vector:** four sections move up to rank 1:
  `exact-basic-logs` (from 3), `exact-gcp-coverage-figure` (from 2), one section of
  `tracing-use-cases` and one section of `gcp-azure-metrics-differences` (each from 2). Others move
  down: the other `gcp-azure-metrics-differences` section (1 to 2), `exact-tag-values` (1 to 2),
  `tagging-standard` (2 to 4), `conflict-commitment-tier-after-spike` (3 to 4) and
  `exact-tag-enforcement` (5 to 6). The net is +2 at @1 and -2 at @3. No case changes between pass
  and fail.
- **Keyword can't decline.** It has no threshold, so it returns chunks for 5 of the 6
  unanswerable questions. It passed the 6th, `unanswerable-subscription-owner`, only because
  all six chunks it returned came from the tagging standard. That case allows the tagging
  standard as related context. Rejection in vector and hybrid comes from the cosine cutoff. The
  held-out `heldout-unanswerable-escalation-policy` question is kept by both (distance 0.592,
  within 0.6). This is a known generalisation gap, unrelated to the keyword/vector choice.

## Generated-answer results (measured separately)

This was one run of `python evals.py --with-llm --output ...` in **vector mode only**, from
16:21 to 16:24 UTC on 8 November 2026, at commit `4767f80`. The run artifact reports
`git_dirty: true` because `evals.py` was edited in the working tree while the run was going.
That edit only changes how a distance is printed in retrieval mode. The loaded code was
`4767f80`. Hybrid and keyword generation were **not measured**.

| | Main set | Held-out | Total |
|---|---|---|---|
| Cases passing (sources, facts, citations, declines) | 37/38 | 8/8 | 45/46 |
| Provider calls (`claude-sonnet-5-5`, no fallbacks) | 38 | 8 | 46 |
| Input tokens | 98,167 | 18,929 | 117,096 |
| Output tokens | 16,455 | 1,804 | 18,259 |
| Cost at list price | $0.36 | $0.06 | **$0.42** |
| Median time per case (retrieval + generation) | 3.6 s | 3.5 s | |

- **Pricing assumption:** Anthropic first-party list price for Claude Sonnet 5.5, $2 per million
  input tokens and $10 per million output tokens. Checked on 8 November 2026 against the pricing
  table dated 2026-10-06. There was no prompt caching (0 cache tokens) and no batch discount. The
  cost is computed from the token counts the provider reported, not from a bill.
- **The one failure** is `unanswerable-logging-spend-amount`. The model declined correctly, then
  suggested running the runbook's query "for project-bravo on 23 April". The decline check flags
  that as an unsupported specific. All the right chunks were retrieved, so this is a generation
  or decline-heuristic issue, not a retrieval one. Retrieval changes would not fix it.
- **Per call:** about 2,550 input and 400 output tokens, or about $0.009.
- **Latency:** warm retrieval takes about 7 ms of the roughly 3.6 s an answer takes. Generation
  is over 99% of answer latency.

## Recommendation for #8

Based only on the numbers above:

1. **Keep vector retrieval over a keyword-only baseline.** Vector passes 32/32 cases against
   30/32. It retrieves 37/37 supporting sections in the top 6 against 35/37, and 35/37 in the
   top 3 against 30/37. Its distance cutoff rejects 6/6 unanswerable questions; keyword rejects
   1/6. In exchange it costs about 7 ms more per warm query and about 2.7 s more at cold start.
   On this corpus that is worth it.
2. **Don't make hybrid the default, and don't build more hybrid work now.** Hybrid ties vector
   on every pass/fail measure, on recall @6 (37/37) and on rejection (6/6). It gains 2 sections
   at @1, loses 2 at @3, raises MRR by 0.027 and adds about 1.6 ms per warm query. That difference
   is too small to separate from noise across 37 sections. Hybrid stays available as an opt-in
   (`--retrieval hybrid`).
3. **Don't implement reranking (#8) on this corpus yet.** A reranker can only reorder chunks that
   were already retrieved. Vector already retrieves every supporting section in the top 6, and
   generation passed every source, fact and citation check. The one generated-answer failure is
   a decline-wording issue that ranking can't fix. There is no measured headroom for a reranker
   to recover.
4. **When to revisit:** reopen #8 when a retrieval eval fails in vector mode, when recall @6 drops
   below the case count, or when the corpus grows enough to rerun this benchmark meaningfully. The
   current index has 82 chunks; the hybrid code itself suggests a few thousand as the scale where
   in-memory scoring needs rethinking.

**Limits:** the evidence covers one small, fictional corpus. The main question set was
written alongside the retrieval tuning, so only the 8 held-out questions are unbiased. There was
one generation run, in one mode. Latency was measured on one laptop. Human search was not
measured.

## Human comparison: not yet measured

The issue also asks how people do with **manual document search** on the same questions,
measured by answer correctness and task completion time. An automated run can't produce that,
and no human results have been estimated or simulated here. The protocol and empty results table
below are for someone to fill in.

### Protocol

1. **Participants:** at least 3 people who know observability but have not read this corpus.
   Record each person's role, not their name.
2. **Questions:** the 32 answerable and 6 unanswerable main-set questions in `evals.EVALS` and
   `evals.UNANSWERABLE_EVALS`. To keep the session short, use a fixed random subset of at least
   10, seeded and recorded, with at least 2 unanswerable questions. Use the same subset in every
   condition.
3. **Conditions**, each participant in a counterbalanced order, with a different question subset
   per condition so nobody answers the same question twice:
   - **A. Manual search:** the `data/` folder in an editor, with file browsing and in-file or
     folder text search allowed. No RAG tools.
   - **B. Retrieval only:** the Streamlit **Find sources** view (vector mode), with no generated
     answer.
   - **C. Generated answer:** **Generate answer** (vector mode, `claude-sonnet-5-5`). The
     participant may check the cited sources.
4. **Timing:** start the timer when the question is shown and stop it when the participant
   submits an answer and the document and section it came from, or says "not in the documents".
   Use a 5-minute limit per question; record a timeout as incorrect at 300 s.
5. **Scoring:** use the same criteria as the automated checks. An answer is **correct** if it
   states every `expected_facts` item and names an `expected_sources` document. An unanswerable
   question is correct only if the participant declines. Two scorers grade answers blind to the
   condition and resolve disagreements; record their agreement.
6. **Record:** per participant, question and condition: correct (yes/no), seconds, the cited
   document and section, and any notes. Report the median time and accuracy per condition, and
   say whether the RAG conditions are faster or more accurate than manual search.

### Results (not yet measured)

| Condition | Participants | Questions each | Accuracy | Median time (s) | p90 time (s) | Unanswerable declined | Notes |
|---|---|---|---|---|---|---|---|
| A. Manual search | not yet measured | | | | | | |
| B. Retrieval only | not yet measured | | | | | | |
| C. Generated answer | not yet measured | | | | | | |

## Reproduce

```bash
python ingest.py
python benchmark.py --output docs/benchmarks/retrieval-$(date +%F).json   # ~1 minute, free
python evals.py --with-llm --output eval-runs/vector.json                 # ~$0.40, needs ANTHROPIC_API_KEY
python evals.py --with-llm --retrieval hybrid --output eval-runs/hybrid.json
```

`benchmark.py --repeats N --cold-runs N` changes how many warm passes and cold starts are run.
