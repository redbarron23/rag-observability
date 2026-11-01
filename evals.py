#!/usr/bin/env python3
"""Eval harness for the observability RAG system.

Tests that the RAG system correctly retrieves the right documents
for known questions over the observability corpus, and declines
questions the corpus can't answer.  Runs in two modes:

  Mode 1 — Retrieval-only (default):
      Checks that the correct documents and supporting sections appear in
      the top-N retrieved chunks, and that nothing survives the distance cutoff
      for unanswerable questions (beyond allowed related context that never
      mentions the subject).  No LLM API key required.

  Mode 2 — Full (--with-llm):
      Also asks the configured LLM to generate an answer and checks that the
      answer contains the expected facts and cites the right sources, and
      that unanswerable questions are declined.  An LLM error counts as a
      failure.

The retrieval-only mode is fast, free, and catches most regressions.
Use --with-llm for the full attribution + fact-presence + citation check.

The answer checks are heuristics: they catch swapped fact pairs, cited
sections that weren't retrieved, and a decline followed by an unsupported
name or number, but passing them is not proof of factual correctness.
Held-out questions are reported separately and don't affect the exit code.

--output PATH also writes a versioned JSON run artifact (configuration,
per-case outcomes, timings, token usage) for comparing runs; see
eval_artifact.py and "Comparing eval runs" in docs/guide.md.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import eval_artifact
from eval_artifact import redact
from query import DEFAULT_RETRIEVAL, N_RESULTS, RETRIEVAL_MODES, RAGEngine

EVALS = [
    {
        "id": "coverage-targets-tier1",
        "question": "What is the coverage target for Tier 1 production resources?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Tier 1")],
        "expected_facts": ["85%", "4 months", "100%", "9 months"],
        # Each target must be stated with its own deadline (issue #29).
        "expected_relations": [("85%", "4 months"), ("100%", "9 months")],
    },
    {
        "id": "coverage-targets-gcp",
        "question": "What is the current GCP coverage level and what is the interim target?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Current status")],
        "expected_facts": ["0.47%", "40%"],
    },
    {
        "id": "framework-scoring",
        "question": "How does the observability framework scoring model work?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_sections": [("architecture/observability_framework.md", "Scoring model")],
        "expected_facts": ["Missing", "Basic", "Good", "Excellent"],
    },
    {
        "id": "framework-categories",
        "question": "What categories are assessed in the observability framework?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_sections": [("architecture/observability_framework.md", "Categories")],
        "expected_facts": ["Metrics", "Logs", "Traces", "Alerts", "SLO", "Ownership"],
    },
    {
        "id": "gcp-azure-metrics-differences",
        "question": "How does GCP metric collection differ from Azure in terms of OpenTelemetry support and native agents?",
        "expected_sources": ["reference/gcp-azure-metrics-comparison.md"],
        "expected_sections": [
            ("reference/gcp-azure-metrics-comparison.md", "OpenTelemetry support"),
            ("reference/gcp-azure-metrics-comparison.md", "Native agents"),
        ],
        "expected_facts": ["Ops Agent", "Azure Monitor Agent"],
    },
    {
        "id": "tagging-standard",
        "question": "What is the resource tagging standard?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Required tags")],
        "expected_facts": ["owner", "cost-centre", "environment"],
    },
    {
        "id": "tracing-use-cases",
        "question": "What are some use cases for distributed tracing?",
        "expected_sources": ["architecture/observability_tracing_use_cases.md"],
        "expected_sections": [
            ("architecture/observability_tracing_use_cases.md", "Latency investigation"),
            ("architecture/observability_tracing_use_cases.md", "Dependency mapping"),
        ],
        "expected_facts": ["dependency", "latency"],
    },
    {
        "id": "data-first-approach",
        "question": "How does the data-first approach avoid unnecessary cloud API calls by checking data freshness first?",
        "expected_sources": ["data-first-approach.md"],
        "expected_sections": [("data-first-approach.md", "Freshness check")],
        "expected_facts": ["6 hours"],
    },
    {
        "id": "observability-setup",
        "question": "According to the observability setup primer, what are the prerequisites and steps for onboarding a new service?",
        "expected_sources": ["observability-setup-primer.md"],
        "expected_sections": [
            ("observability-setup-primer.md", "Prerequisites"),
            ("observability-setup-primer.md", "Onboarding steps"),
        ],
        "expected_facts": ["owning team", "SLO"],
    },
    {
        "id": "cost-smoothing",
        "question": "What is the cost smoothing plan for Log Analytics?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_sections": [("programme/COST_SMOOTHING.md", "Plan")],
        "expected_facts": ["20%", "three waves"],
    },
    # Exact-term questions: figures, product names and tag names that
    # MiniLM embeddings match less reliably than keywords (issue #8).
    {
        "id": "exact-gcp-coverage-figure",
        "question": "Which cloud is at 0.47% coverage?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Current status")],
        "expected_facts": ["GCP"],
    },
    {
        "id": "exact-ops-agent",
        "question": "What does the Ops Agent collect and how is it configured?",
        "expected_sources": ["reference/gcp-azure-metrics-comparison.md"],
        "expected_sections": [("reference/gcp-azure-metrics-comparison.md", "Native agents")],
        "expected_facts": ["host metrics", "YAML"],
    },
    {
        "id": "exact-tag-values",
        "question": "What values are allowed for the environment and tier tags?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Allowed values")],
        "expected_facts": ["prod", "staging", "dev"],
    },
    {
        "id": "exact-tag-enforcement",
        "question": "What happens to resources missing the owner or cost-centre tag?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Enforcement")],
        "expected_facts": ["Azure Policy", "30 days"],
    },
    {
        "id": "exact-basic-logs",
        "question": "Which Log Analytics tables move to the Basic Logs tier?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_sections": [("programme/COST_SMOOTHING.md", "Plan")],
        "expected_facts": ["high-volume", "30 days"],
    },
    # Short questions about a named thing embed poorly: the right chunk ranks
    # first but sits just past the distance cutoff (~0.64). They pass because
    # the chunk contains the exact term (query.EXACT_TERM_MARGIN).
    {
        "id": "short-ops-agent",
        "question": "What does the Ops Agent do?",
        "expected_sources": ["reference/gcp-azure-metrics-comparison.md"],
        "expected_sections": [("reference/gcp-azure-metrics-comparison.md", "Native agents")],
        "expected_facts": ["host metrics", "YAML"],
    },
    {
        "id": "short-last-updated",
        "question": "What is last_updated used for?",
        "expected_sources": ["data-first-approach.md"],
        "expected_sections": [("data-first-approach.md", "Freshness check")],
        "expected_facts": ["6 hours"],
    },
    # Cross-repo questions from the obs-agent label review (issue #56): the
    # shared team/env/cost-centre vocabulary and what obs-agent leaves unchecked.
    {
        "id": "obs-agent-suspect-env",
        "question": "Why is this environment label suspect, and what does our standard require?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [
            ("reference/tagging-standard.md", "Allowed values"),
            ("reference/tagging-standard.md", "Environment vocabulary"),
        ],
        "expected_facts": ["prod", "staging", "dev", "sandbox"],
    },
    {
        "id": "obs-agent-env-aliases",
        "question": "Which canonical environment do the legacy values prd and uat map to?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Environment vocabulary")],
        "expected_facts": ["prod", "staging"],
    },
    {
        "id": "obs-agent-owner-meaning",
        "question": "Does the owner tag name a person or a team?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Owner")],
        "expected_facts": ["team", "not a person"],
    },
    {
        "id": "obs-agent-team-cost-centre",
        "question": "Which cost centre belongs to the data-platform team, and what legacy spellings of that team are accepted?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Team vocabulary")],
        "expected_facts": ["CC-2002", "dp"],
    },
    {
        "id": "obs-agent-field-mapping",
        "question": "How do the policy tag keys map to the obs-agent team, env and cost_center fields?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Field mapping for obs-agent")],
        "expected_facts": ["owner", "environment", "cost-centre"],
    },
    {
        "id": "obs-agent-not-evaluated",
        "question": "Which tagging policy requirements does obs-agent not evaluate?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [
            ("reference/tagging-standard.md", "Requirements obs-agent does not evaluate"),
        ],
        "expected_facts": ["tier", "service", "resource-level", "enforcement"],
    },
    {
        "id": "obs-agent-tag-levels",
        "question": "Do tags go on each resource or only on the subscription or project?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Tag levels")],
        "expected_facts": ["not inherited"],
    },
    {
        "id": "label-completeness-vs-coverage",
        "question": "Is label completeness the same as metrics, logs and traces coverage?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Label completeness and coverage")],
        "expected_facts": ["neither substitutes"],
    },
]

# Questions on observability-adjacent topics that the sample corpus doesn't
# cover. The right answer is to decline, not to invent a rota or a price.
UNANSWERABLE_EVALS = [
    {
        "id": "unanswerable-oncall-rota",
        "question": "Who is on the on-call rota for the payments team this week?",
    },
    {
        "id": "unanswerable-vendor-pricing",
        "question": "How much does a Datadog Pro licence cost per host per month?",
    },
    {
        "id": "unanswerable-k8s-version",
        "question": "Which Kubernetes version does the production AKS cluster run?",
    },
    {
        "id": "unanswerable-outage-root-cause",
        "question": "What was the root cause of last month's checkout outage?",
    },
    # The tagging standard lists the allowed owner values but never says which
    # team owns a given subscription, so the owner must stay unknown (issue
    # #56). Policy chunks are related and may be retrieved; the retrieval check
    # passes if only allowed_sources survive the cutoff and none of them
    # mentions the resource (absent_terms). --with-llm still requires a decline.
    {
        "id": "unanswerable-subscription-owner",
        "question": "Which team owns the Azure subscription az-demo-007?",
        "allowed_sources": ["reference/tagging-standard.md"],
        "absent_terms": ["az-demo-007"],
    },
]

# Held-out questions (issue #29). Written on 2026-11-01 after the distance
# cutoff (0.6) and exact-term margin (0.1) were fixed, about sections no other
# case targets. They are not used to tune retrieval: their results are
# reported separately and don't affect the exit code, so a failure here is a
# generalisation signal to record, not a reason to retune the cutoff.
HELDOUT_EVALS = [
    {
        "id": "heldout-tier2-targets",
        "question": "What coverage targets apply to internal production workloads?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Tier 2")],
        "expected_facts": ["70%", "6 months", "95%", "12 months"],
        "expected_relations": [("70%", "6 months"), ("95%", "12 months")],
    },
    {
        "id": "heldout-alert-noise",
        "question": "How much alert noise is acceptable for an Excellent alerts score?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_sections": [("architecture/observability_framework.md", "Alerts")],
        "expected_facts": ["non-actionable", "per week"],
    },
    {
        "id": "heldout-error-propagation",
        "question": "How does tracing help when an error shows up at the edge?",
        "expected_sources": ["architecture/observability_tracing_use_cases.md"],
        "expected_sections": [
            ("architecture/observability_tracing_use_cases.md", "Error propagation"),
        ],
        "expected_facts": ["failing component", "mean time to resolution"],
    },
    {
        "id": "heldout-commitment-tier",
        "question": "When does the programme buy a Log Analytics commitment tier, and how large is it?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_sections": [("programme/COST_SMOOTHING.md", "Commitment tiers")],
        "expected_facts": ["two consecutive months", "100 GB per day"],
    },
    {
        "id": "heldout-cost-centre-format",
        "question": "How are cost-centre values written on GCP compared with Azure?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Cost centre")],
        "expected_facts": ["lowercase", "cc-1001", "CC-1001"],
    },
    {
        "id": "heldout-reporting-principle",
        "question": "Why do reporting jobs read the central metrics store before calling cloud APIs?",
        "expected_sources": ["data-first-approach.md"],
        "expected_sections": [("data-first-approach.md", "Principle")],
        "expected_facts": ["throttling", "reproducible"],
    },
]

HELDOUT_UNANSWERABLE_EVALS = [
    {
        "id": "heldout-unanswerable-escalation-policy",
        "question": "Which PagerDuty escalation policy does the checkout service use?",
    },
    {
        "id": "heldout-unanswerable-grafana-version",
        "question": "Which Grafana version runs the central dashboards?",
    },
]


def contains_fact(answer: str, fact: str) -> bool:
    """Case-insensitive match that must start at a word boundary, so that
    "4 months" doesn't match "14 months" and "5%" doesn't match "85%".
    The end is left open so "SLO" still matches "SLOs"."""
    return re.search(r"(?<!\w)" + re.escape(fact), answer, re.IGNORECASE) is not None


# Phrases that signal the answer declines rather than answering. Includes the
# engine's own replies when nothing is retrieved or the model refuses.
DECLINE_RE = re.compile(
    r"\b(?:does not|doesn't|do not|don't) (?:contain|include|mention|cover|provide|specify|say|have)"
    r"|\bnot (?:in|included in|contained in|covered (?:in|by)|mentioned in|available in|present in)"
    r" the (?:provided |given |retrieved )?(?:context|documents?|sources?|chunks?)"
    r"|\bno (?:information|mention|details?|data)\b"
    r"|\b(?:cannot|can't|unable to) (?:answer|determine|find|tell)"
    r"|\bno relevant documents found"
    r"|\bthe model declined",
    re.IGNORECASE,
)


def is_decline(answer: str) -> bool:
    """True if the answer says the context doesn't hold the answer."""
    return DECLINE_RE.search(answer) is not None


# A citation parenthesis, e.g. "(source: reference/coverage-targets.md, section '…')".
# Models sometimes list several documents in one: "(source: a.md; b.md)".
CITATION_RE = re.compile(r"\(\s*sources?:[^)]*\)", re.IGNORECASE)
DOCUMENT_RE = re.compile(r"[\w./-]+\.md\b")


def cited_sources(answer: str) -> set[str]:
    """Documents the answer cites with the (source: X.md, ...) format."""
    return {doc for citation in CITATION_RE.findall(answer)
            for doc in DOCUMENT_RE.findall(citation)}


def same_document(a: str, b: str) -> bool:
    """Whether two source paths name the same document, allowing either one
    to drop leading directories ("coverage-targets.md" vs
    "reference/coverage-targets.md")."""
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def citation_problems(answer: str, retrieved_sources: set[str],
                      expected_sources: list[str],
                      citations: list[dict] | None = None) -> list[str]:
    """Return what's wrong with the answer's citations; empty if they're fine.

    Uses the structured citations when the provider returned them (Anthropic),
    otherwise parses the citations written in the answer text (DeepSeek).
    Fails when the answer cites nothing, cites a document that wasn't
    retrieved (so the citation can't be backed by the context), or never
    cites the document the answer should come from.
    """
    if citations is not None:
        cited = {c["source"] for c in citations}
    else:
        cited = cited_sources(answer)
    if not cited:
        return ["no citations"]
    problems = []
    unretrieved = sorted(c for c in cited
                         if not any(same_document(c, r) for r in retrieved_sources))
    if unretrieved:
        problems.append(f"cites sources that weren't retrieved: {unretrieved}")
    uncited = [e for e in expected_sources
               if not any(same_document(c, e) for c in cited)]
    if uncited:
        problems.append(f"doesn't cite expected sources: {uncited}")
    return problems


def unanswerable_problems(chunks: list[dict], ev: dict) -> list[str]:
    """Why the chunks kept for an unanswerable question could support an
    answer; empty if they can't."""
    allowed = set(ev.get("allowed_sources", []))
    problems = [f"kept {c['source']}" for c in chunks if c["source"] not in allowed]
    problems += [f"{c['source']} mentions {term!r}"
                 for c in chunks if c["source"] in allowed
                 for term in ev.get("absent_terms", [])
                 if contains_fact(c.get("text", ""), term)]
    return problems


# ── Stricter answer checks (issue #29) ─────────────────────────────────
# These are offline heuristics. They catch specific failure shapes (swapped
# fact pairs, invented sections, a decline followed by a guess); passing them
# is not proof that an answer is factually correct.

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]


def _fact_spans(text: str, fact: str) -> list[tuple[int, int]]:
    return [m.span() for m in
            re.finditer(r"(?<!\w)" + re.escape(fact), text, re.IGNORECASE)]


def _stated_together(sentence: str, first: str, second: str, others: set[str]) -> bool:
    for _, first_end in _fact_spans(sentence, first):
        later = [start for start, _ in _fact_spans(sentence, second) if start >= first_end]
        if not later:
            continue
        between = sentence[first_end:min(later)]
        if not any(_fact_spans(between, other) for other in others):
            return True
    return False


def relation_problems(answer: str, relations: list[tuple[str, str]]) -> list[str]:
    """Pairs of facts that the answer doesn't state together.

    A pair (first, second) holds when, within one sentence, first is followed
    by second with no other fact from the case's pairs in between. "85% within
    4 months and 100% within 9 months" satisfies (85%, 4 months) and
    (100%, 9 months); the swapped "100% within 4 months and 85% within
    9 months" satisfies neither. The check is order-sensitive, so a correct
    answer phrased the other way round ("within 4 months, 85%") also fails:
    it prefers a false failure to a false pass.
    """
    terms = {term for pair in relations for term in pair}
    return [f"{first!r} not stated with {second!r}"
            for first, second in relations
            if not any(_stated_together(s, first, second, terms - {first, second})
                       for s in sentences(answer))]


SECTION_RE = re.compile(r"sections?\s*:?\s*['\"\u2018\u201c](.+?)['\"\u2019\u201d]", re.IGNORECASE)
BREADCRUMB_RE = re.compile(r"\s*[>\u203a\u00bb]\s*")


def normalise_section(section: str) -> str:
    """Normalisation for comparing a cited section with chunk metadata:
    keep only the last heading of a breadcrumb ("Coverage Targets > Tier 1"
    -> "Tier 1"), drop surrounding quotes and trailing punctuation, collapse
    whitespace and ignore case. Anything else must match exactly."""
    last = BREADCRUMB_RE.split(section)[-1]
    last = last.strip().strip("'\"\u2018\u2019\u201c\u201d").rstrip(".,;:").strip()
    return " ".join(last.split()).casefold()


def cited_sections(answer: str) -> list[tuple[str, str]]:
    """(document, section) pairs from written citations such as
    "(source: a.md, section 'X'; source: b.md, section 'Y')"."""
    pairs = []
    for citation in CITATION_RE.findall(answer):
        for part in citation.strip("()").split(";"):
            document, section = DOCUMENT_RE.search(part), SECTION_RE.search(part)
            if document and section:
                pairs.append((document.group(0), section.group(1)))
    return pairs


def section_problems(answer: str, chunks: list[dict],
                     citations: list[dict] | None = None) -> list[str]:
    """Cited sections that no retrieved chunk carries.

    Checks the structured citations when the provider returned them, and the
    sections written in the answer text in either case (on the Anthropic path
    those markers are generated from chunk metadata). A citation without a
    section is not checked here; citation_problems covers the document.
    """
    cited = cited_sections(answer)
    if citations:
        cited += [(c["source"], c.get("section") or "") for c in citations
                  if c.get("section")]
    retrieved = [(c["source"], normalise_section(c.get("section") or "")) for c in chunks]
    problems = []
    for document, section in dict.fromkeys(cited):
        if not any(same_document(document, source) and normalise_section(section) == name
                   for source, name in retrieved):
            problems.append(f"cites a section that wasn't retrieved: {document} / {section!r}")
    return problems


NEGATION_RE = re.compile(
    r"\b(?:no|not|none|neither|nor|never|cannot|without|unable|unknown)\b|n't\b",
    re.IGNORECASE,
)
# Pointers elsewhere ("Check your paging tool") are not answers.
ADVICE_RE = re.compile(
    r"^\s*(?:you (?:may|might|could|should|can|would)|please|try|check|consult|ask"
    r"|contact|refer|see|look|consider|i (?:recommend|suggest))\b",
    re.IGNORECASE,
)
SPECIFIC_RE = re.compile(r"[\w.-]*\d[\w.-]*|\b[A-Z][\w-]*")
WORD_RE = re.compile(r"[\w.-]+")
# Capitalised only because they start a sentence.
SENTENCE_STARTERS = frozenset("""
    a an the this that these those they it its there here i we you he she my our your their
    however but and so also only each all both some any based according in on for to as if
    while although instead unfortunately therefore because what which who when where why how
    note answer documents document context sources source information
""".split())


def unsupported_specifics(sentence: str, known: set[str]) -> list[str]:
    """Names and numbers in a sentence that appear in neither the question
    nor the corpus, e.g. "Alice" or "1.29"."""
    found = []
    for token in SPECIFIC_RE.findall(sentence):
        word = token.strip(".-")
        if word and word.lower() not in known and word.lower() not in SENTENCE_STARTERS:
            found.append(word)
    return found


def decline_problems(answer: str, question: str, corpus_text: str) -> list[str]:
    """Why an answer to an unanswerable question is not a clean decline.

    The answer must decline (is_decline), and no sentence may add a name or
    number found in neither the question nor the corpus, unless the sentence
    is negated ("None of them mention AKS") or points elsewhere ("Check your
    paging tool"). This catches "The context does not contain a rota. Alice
    is on call this week." It does not catch a guess built only from ordinary
    words or from terms that happen to occur in the corpus.
    """
    if not is_decline(answer):
        return ["does not decline"]
    known = {w.strip(".-").lower() for w in WORD_RE.findall(f"{question} {corpus_text}")}
    problems = []
    for sentence in sentences(answer):
        if NEGATION_RE.search(sentence) or ADVICE_RE.search(sentence):
            continue
        extra = unsupported_specifics(sentence, known)
        if extra:
            problems.append(f"unsupported claim alongside the decline: {sentence!r} "
                            f"({', '.join(extra)})")
    return problems


def corpus_text(engine) -> str:
    """Every indexed chunk's text, or "" when the engine has no collection."""
    try:
        return "\n".join(engine.collection.get(include=["documents"])["documents"] or [])
    except AttributeError:
        return ""


RETRIEVE_N = 6  # number of chunks to retrieve for document and section checks


def missing_evidence(chunks: list[dict], facts: list[str]) -> list[str]:
    """Expected facts that appear in none of the retrieved chunks' text.

    Every expected fact is a literal string from the source section, so a
    model can only state it if retrieval delivered it."""
    return [f for f in facts if not any(contains_fact(c.get("text", ""), f) for c in chunks)]


def supporting_section_ranks(chunks: list[dict],
                             expected_sections: list[tuple[str, str]]) -> list[int | None]:
    """First 1-based rank for each required source/section, or None if absent."""
    return [
        next((rank for rank, chunk in enumerate(chunks, start=1)
              if (chunk["source"], chunk.get("section")) == (source, section)), None)
        for source, section in expected_sections
    ]


def case_record(ev: dict, kind: str, ok: bool, dimensions: dict, problems: list[str],
                started: float, **extra) -> dict:
    """One case's outcome for the run artifact (issue #39): no document text,
    and no answer unless the caller adds it."""
    return {"id": ev["id"], "kind": kind, "status": "pass" if ok else "fail",
            "dimensions": dimensions, "problems": [redact(p) for p in problems],
            "duration_seconds": round(time.perf_counter() - started, 3), **extra}


def run_retrieval_evals(engine: RAGEngine, verbose: bool,
                        answerable: list[dict] | None = None,
                        unanswerable: list[dict] | None = None,
                        records: list[dict] | None = None) -> tuple[int, int]:
    """Run retrieval-only evals — no LLM API call needed.

    Each case's outcome is appended to records when it is given."""
    answerable = EVALS if answerable is None else answerable
    unanswerable = UNANSWERABLE_EVALS if unanswerable is None else unanswerable
    records = [] if records is None else records
    passed = 0
    failed = 0
    source_passed = 0
    section_hits = dict.fromkeys((1, 3, RETRIEVE_N), 0)
    section_total = 0
    evidence_failures = 0

    print(f"Retrieval evals (documents and supporting sections in top-{RETRIEVE_N})")
    print("=" * 60)

    for ev in answerable:
        started = time.perf_counter()
        chunks = engine.retrieve(ev["question"], n_results=RETRIEVE_N)
        retrieved_sources = {c["source"] for c in chunks}

        # Check that all expected sources appear in the retrieved set
        expected = ev["expected_sources"]
        missing = [s for s in expected if s not in retrieved_sources]
        sources_ok = not missing
        source_passed += sources_ok
        ranks = supporting_section_ranks(chunks, ev["expected_sections"])
        section_total += len(ranks)
        for cutoff in section_hits:
            section_hits[cutoff] += sum(rank is not None and rank <= cutoff for rank in ranks)
        sections_ok = all(rank is not None and rank <= RETRIEVE_N for rank in ranks)
        first_rank = min((rank for rank in ranks if rank is not None), default=None)
        # Any chunk labelled with the right section satisfies the section
        # check, so also require the facts themselves in the retrieved text
        # (issue #72: a split section can return the wrong half).
        unsupported = missing_evidence(chunks, ev.get("expected_facts", []))
        evidence_ok = not unsupported
        evidence_failures += not evidence_ok
        ok = sources_ok and sections_ok and evidence_ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}  sources={'ok' if sources_ok else 'FAIL'} "
              f"sections={'ok' if sections_ok else 'FAIL'} "
              f"evidence={'ok' if evidence_ok else 'FAIL'} "
              f"first relevant rank={first_rank if first_rank is not None else 'missing'}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing:
                print(f"       Missing sources: {missing}")
            for (source, section), rank in zip(ev["expected_sections"], ranks, strict=True):
                print(f"       {source} / {section}: rank {rank if rank is not None else 'missing'}")
            if unsupported:
                print(f"       Facts not in retrieved text: {unsupported}")

        records.append(case_record(
            ev, "answerable", ok,
            {"sources": sources_ok, "sections": sections_ok, "evidence": evidence_ok},
            ([f"missing sources: {missing}"] if missing else [])
            + [f"section not in top-{RETRIEVE_N}: {source} / {section}"
               for (source, section), rank in zip(ev["expected_sections"], ranks, strict=True)
               if rank is None or rank > RETRIEVE_N]
            + ([f"facts not in retrieved text: {unsupported}"] if unsupported else []),
            started,
            retrieved_sources=sorted(retrieved_sources),
            section_ranks=[{"source": source, "section": section, "rank": rank}
                           for (source, section), rank
                           in zip(ev["expected_sections"], ranks, strict=True)],
            first_relevant_rank=first_rank,
        ))

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nDocument attribution: {source_passed}/{len(answerable)} cases")
    print(f"Fact evidence in retrieved text: {len(answerable) - evidence_failures}/"
          f"{len(answerable)} cases")
    print("Supporting-section recall: " + ", ".join(
        f"@{cutoff} {hits}/{section_total}" for cutoff, hits in section_hits.items()))

    # Unanswerable questions: nothing should survive the distance cutoff,
    # except related context from allowed_sources that holds no evidence
    # about the subject (none of absent_terms).
    for ev in unanswerable:
        started = time.perf_counter()
        chunks = engine.retrieve(ev["question"], n_results=RETRIEVE_N)
        rejected = unanswerable_problems(chunks, ev)
        ok = not rejected
        records.append(case_record(
            ev, "unanswerable", ok, {"rejection": ok}, rejected, started,
            kept=[{"source": c["source"], "distance": c.get("distance"),
                   "allowed": c["source"] in ev.get("allowed_sources", [])} for c in chunks],
        ))

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            for c in chunks:
                allowed = " (allowed)" if c["source"] in ev.get("allowed_sources", []) else ""
                print(f"       Kept: {c['source']} (dist {c['distance']:.3f}){allowed}")

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def generation_record(engine) -> dict:
    """A copy of the engine's metadata for its last provider call."""
    meta = getattr(engine, "last_generation", None)
    if not isinstance(meta, dict):
        return {"llm_called": None,
                "unavailable_reason": "the engine did not report generation metadata"}
    meta = dict(meta)
    if meta.get("llm_called") and meta.get("usage") is None:
        meta["usage_unavailable_reason"] = "the provider response had no token usage"
    return meta


def run_full_evals(engine: RAGEngine, verbose: bool,
                   answerable: list[dict] | None = None,
                   unanswerable: list[dict] | None = None,
                   records: list[dict] | None = None,
                   include_answers: bool = False) -> tuple[int, int]:
    """Run full evals — retrieval + LLM answer + fact and citation checks.

    Each case's outcome is appended to records when it is given; answers are
    only recorded with include_answers."""
    answerable = EVALS if answerable is None else answerable
    unanswerable = UNANSWERABLE_EVALS if unanswerable is None else unanswerable
    records = [] if records is None else records
    corpus = corpus_text(engine)

    def error_record(ev, kind, dimensions, error, started):
        return case_record(ev, kind, False, dict.fromkeys(dimensions), [], started,
                           status="error",
                           error={"type": type(error).__name__, "message": redact(str(error))},
                           generation=generation_record(engine))

    def answer_fields(answer):
        return {"answer": redact(answer)} if include_answers else {}
    passed = 0
    failed = 0
    dimension_failures = {"sources": 0, "facts": 0, "citations": 0, "declines": 0}
    answerable_errors = 0
    unanswerable_errors = 0

    print("\nFull evals (retrieval + LLM answer + fact presence + citations)")
    print("=" * 60)

    for ev in answerable:
        started = time.perf_counter()
        try:
            answer, chunks, citations = engine.answer_with_citations(
                ev["question"], n_results=N_RESULTS, verbose=False,
            )
        except Exception as e:  # noqa: BLE001 — any provider/SDK error fails the case
            print(f"[FAIL] {ev['id']} — LLM error: {e}")
            records.append(error_record(ev, "answerable", ("sources", "facts", "citations"),
                                        e, started))
            failed += 1
            answerable_errors += 1
            continue

        # Check 1: source attribution in retrieved chunks
        retrieved_sources = {c["source"] for c in chunks}
        expected = ev["expected_sources"]
        found = [s for s in expected if any(s in r for r in retrieved_sources)]
        missing_sources = [s for s in expected if s not in found]
        sources_ok = len(missing_sources) == 0

        # Check 2: fact presence in the answer
        missing_facts = [f for f in ev["expected_facts"] if not contains_fact(answer, f)]
        unrelated = relation_problems(answer, ev.get("expected_relations", []))
        facts_ok = not missing_facts and not unrelated

        # Check 3: the answer's citations (structured when the provider has them)
        bad_citations = (citation_problems(answer, retrieved_sources, expected, citations)
                         + section_problems(answer, chunks, citations))
        citations_ok = not bad_citations

        dimension_failures["sources"] += not sources_ok
        dimension_failures["facts"] += not facts_ok
        dimension_failures["citations"] += not citations_ok

        ok = sources_ok and facts_ok and citations_ok
        status = "PASS" if ok else "FAIL"
        dims = " ".join(f"{name}={'ok' if good else 'FAIL'}" for name, good in
                        [("sources", sources_ok), ("facts", facts_ok),
                         ("citations", citations_ok)])
        print(f"[{status}] {ev['id']}  {dims}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing_sources:
                print(f"       Missing sources: {missing_sources}")
            if missing_facts:
                print(f"       Missing facts: {missing_facts}")
            for problem in unrelated:
                print(f"       Facts: {problem}")
            for problem in bad_citations:
                print(f"       Citations: {problem}")
            print(f"       A: {answer[:300]}")

        records.append(case_record(
            ev, "answerable", ok,
            {"sources": sources_ok, "facts": facts_ok, "citations": citations_ok},
            ([f"missing sources: {missing_sources}"] if missing_sources else [])
            + ([f"missing facts: {missing_facts}"] if missing_facts else [])
            + unrelated + bad_citations,
            started,
            retrieved_sources=sorted(retrieved_sources),
            native_citations=citations is not None,
            generation=generation_record(engine),
            **answer_fields(answer),
        ))

        if ok:
            passed += 1
        else:
            failed += 1

    # Unanswerable questions run without the distance cutoff, so the model
    # sees the nearest chunks and has to decline on its own.
    for ev in unanswerable:
        started = time.perf_counter()
        try:
            answer, chunks = engine.answer(
                ev["question"], n_results=N_RESULTS, verbose=False,
                max_distance=None,
            )
        except Exception as e:  # noqa: BLE001 — any provider/SDK error fails the case
            print(f"[FAIL] {ev['id']} — LLM error: {e}")
            records.append(error_record(ev, "unanswerable", ("declines",), e, started))
            failed += 1
            unanswerable_errors += 1
            continue

        bad_decline = decline_problems(answer, ev["question"],
                                       corpus + "\n" + "\n".join(c.get("text", "") for c in chunks))
        ok = not bad_decline
        dimension_failures["declines"] += not ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}  declines={'ok' if ok else 'FAIL'}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted({c['source'] for c in chunks}))}")
            for problem in bad_decline:
                print(f"       Declines: {problem}")
            print(f"       A: {answer[:300]}")

        records.append(case_record(
            ev, "unanswerable", ok, {"declines": ok}, bad_decline, started,
            retrieved_sources=sorted({c["source"] for c in chunks}),
            generation=generation_record(engine),
            **answer_fields(answer),
        ))

        if ok:
            passed += 1
        else:
            failed += 1

    totals = {"sources": len(answerable), "facts": len(answerable), "citations": len(answerable),
              "declines": len(unanswerable)}
    unevaluated = {"sources": answerable_errors, "facts": answerable_errors,
                   "citations": answerable_errors, "declines": unanswerable_errors}
    print(f"\nProvider errors: {answerable_errors + unanswerable_errors}/{passed + failed}")
    print("\nBy dimension: " + ", ".join(
        f"{name} {totals[name] - unevaluated[name] - dimension_failures[name]}/"
        f"{totals[name] - unevaluated[name]} ({unevaluated[name]} unevaluated)"
        for name in totals))

    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evals")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run LLM-based fact-presence, citation and decline checks")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--persist-dir", default="chroma_db",
                        help="Chroma persistence directory")
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=RETRIEVAL_MODES,
                        help=f"Retrieval mode to evaluate (default: {DEFAULT_RETRIEVAL})")
    parser.add_argument("--output", "-o", metavar="PATH",
                        help="Also write a JSON run artifact (configuration, per-case "
                             "outcomes, timings, token usage) to PATH")
    parser.add_argument("--include-answers", action="store_true",
                        help="With --output and --with-llm, also record each generated "
                             "answer (off by default: answers quote the corpus)")
    args = parser.parse_args()
    if args.include_answers and not args.output:
        parser.error("--include-answers needs --output")

    started_at, started = eval_artifact.utc_now(), time.perf_counter()
    engine = RAGEngine(persist_dir=args.persist_dir, retrieval=args.retrieval)
    print(f"Retrieval mode: {args.retrieval}\n")
    results: dict[str, list[dict]] = {"retrieval": [], "heldout_retrieval": []}

    p1, f1 = run_retrieval_evals(engine, verbose=args.verbose, records=results["retrieval"])
    print(f"\nRetrieval results: {p1}/{p1 + f1} passed")
    print("\nHeld-out questions (reported separately; not used for tuning and "
          "not counted in the exit code)")
    hp1, hf1 = run_retrieval_evals(engine, args.verbose,
                                   HELDOUT_EVALS, HELDOUT_UNANSWERABLE_EVALS,
                                   records=results["heldout_retrieval"])
    print(f"\nHeld-out retrieval: {hp1}/{hp1 + hf1} passed")

    if args.with_llm:
        results["full"], results["heldout_full"] = [], []
        p2, f2 = run_full_evals(engine, verbose=args.verbose, records=results["full"],
                                include_answers=args.include_answers)
        print(f"Full results:       {p2}/{p2 + f2} passed")
        hp2, hf2 = run_full_evals(engine, args.verbose,
                                  HELDOUT_EVALS, HELDOUT_UNANSWERABLE_EVALS,
                                  records=results["heldout_full"],
                                  include_answers=args.include_answers)
        print(f"Held-out full:      {hp2}/{hp2 + hf2} passed")
        print(f"\nCombined: {p1 + p2}/{p1 + p2 + f1 + f2} passed")
        failures = f1 + f2
    else:
        print("(Run with --with-llm to also check facts, citations and declines in generated answers)")
        failures = f1

    exit_code = 1 if failures else 0
    if args.output:
        artifact = eval_artifact.build_artifact(
            engine=engine, mode=args.retrieval, retrieve_n=RETRIEVE_N,
            with_llm=args.with_llm, include_answers=args.include_answers,
            started_at=started_at, finished_at=eval_artifact.utc_now(),
            duration_seconds=time.perf_counter() - started, exit_code=exit_code,
            results=results,
        )
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote run artifact: {path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
