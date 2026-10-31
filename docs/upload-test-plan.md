# Test the Markdown upload flow

Use the three fictional documents in [upload-samples](upload-samples/). They cover service standards, a latency runbook, and a historical incident. Their facts are invented; no credentials, customer records, or real infrastructure details are included.

## Upload and search

1. Open the local Streamlit interface.
2. Under **Add documents**, select the three `.md` files from `docs/upload-samples/`. Do not upload this test plan: its expected answers would contaminate the retrieval test.
3. Click **Upload and index** and wait for the success message. This rebuilds the selected index from `data/`, including existing documents and prior uploads.
4. Leave the defaults at six chunks and a maximum distance of 0.6. Ask the questions below using **Find sources**.
5. Inspect the returned section names and text. Record missing sections rather than changing the expected answer to match retrieval. If a section is missing, temporarily increase the chunk count to 12 and/or use distance 2 to diagnose ranking and filtering; those diagnostic results are separate from the default-settings result.
6. Optionally click **Generate answer** to check factual accuracy and citations. This makes a provider request and may incur charges. Retrieval alone needs no provider key.

## Expected evidence and answers

The filenames below are the source document names; the UI prefixes uploaded files with `uploads/<content hash>/`.

| Question | Required source and section | Expected answer |
| --- | --- | --- |
| Which team owns Cobalt Checkout? | `cobalt-checkout-service-standard.md` → Ownership | Atlas Commerce; service identifier `cobalt-checkout`. |
| What is Cobalt Checkout's availability SLO and measurement window? | `cobalt-checkout-service-standard.md` → Availability SLO | 99.95% over a rolling 28 days. |
| How long are Cobalt Checkout operational logs, debug logs, and traces retained? | `cobalt-checkout-service-standard.md` → Telemetry retention | 30 days, 7 days, and 14 days respectively. |
| What should I inspect when investigating Cobalt Checkout latency? | `cobalt-checkout-latency-runbook.md` → Triage | Inventory and payment spans, correlated logs, connection-pool usage, queued requests, and configuration changes. |
| At what latency threshold should I page Cobalt Checkout's primary on-call? | `cobalt-checkout-latency-runbook.md` → Escalation | p95 above 700 ms for 15 consecutive minutes. The runbook also provides a separate HTTP 5xx escalation condition. |
| What caused the Cobalt Checkout incident on 8 October 2026? | `cobalt-checkout-incident-review.md` → Root cause | Inventory connection-pool limit reduced from 80 to 20, causing queuing and timeouts; not the payment service. |
| How long did the Cobalt Checkout incident on 8 October 2026 last? | `cobalt-checkout-incident-review.md` → Timeline | 19 minutes, from 13:05 to 13:24 UTC. |
| Which team owns Cobalt Checkout, and when should I escalate high latency? | Service standard → Ownership **and** latency runbook → Escalation | Atlas Commerce; page the primary on-call above 700 ms p95 for 15 consecutive minutes. Both source sections must be retrieved. |

A document's introduction is not sufficient evidence when the required section is missing. Generated answers must attach citations to the facts they support; the presence of a familiar number somewhere in the answer is insufficient.

## Initial retrieval check — 17 October 2026

Validated the actual upload-saving and ingestion functions in a temporary index containing the eight original documents plus these three uploads: 11 documents and 55 chunks. With `all-MiniLM-L6-v2`, sentence-transformers 6.1.0, six chunks, and a distance cutoff of 0.6, seven of the eight answerable questions retrieved all required sections.

The cross-document question missed `Ownership`, which ranked seventh with distance approximately 0.408. Its `Escalation` section ranked first. Increasing the chunk count to 12 exposed the ownership evidence, confirming a ranking/top-k limit rather than rejection by the distance cutoff. Keep this case as a known failure at the default settings. The seven single-document questions passed. Generation and the unanswerable-question behavior have not been checked for these drafts.

## Questions the documents cannot answer

- **Which person is on call for Cobalt Checkout today?** The documents name a team and routing process, not a current person. The answer should explain that the rota is not provided.
- **What is Cobalt Checkout's current production connection-pool limit?** The incident review records a historical rollback to 80. The answer must distinguish that history from an unverified current configuration.
- **How much revenue was lost during the 8 October incident?** The review explicitly provides no revenue estimate. The model should decline to invent a number.

Relevant chunks may still be retrieved for these questions. The generation check is whether the answer acknowledges missing evidence and avoids guessing, not whether retrieval returns an empty set.

## Upload behavior checks

- Upload the same files again: the saved document paths should be reused, without duplicate copies.
- Upload a changed copy with the same filename: both contents are retained separately. This is addition, not automatic replacement of a policy. Remove an obsolete saved file and rebuild if testing replacement behavior.
- Try a file containing only `tiny`, or a long heading with no body: validation should fail before indexing. Keep invalid fixtures outside the sample document folder.

These files stay outside `data/` until explicitly uploaded, so they do not change the bundled corpus by themselves. Uploading them can change the existing eval rankings and the meaning of previously unanswerable questions. This is a manual test plan, not an extension of the automated cases in `evals.py`.
