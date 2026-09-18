# PRD: Campus Copilot

**Owner:** Piyush · **Status:** v1 built, pilot pending · **Last updated:** September 2026

---

## 1. Problem

NIT Rourkela students constantly need answers to rule-based questions: minimum attendance, what happens after a missed mid-sem, how CGPA is computed, registration dates, certificate processes, branch change conditions.

The answers exist, but they are hard to reach:

- They are spread across long PDFs (the UG regulations alone run to 31 pages), yearly academic calendars, and one-off notices on nitrkl.ac.in.
- Rules change between years, so a senior's answer from memory is often out of date.
- The fallback is a WhatsApp group or a visit to an office, which is slow for students and repetitive for staff.

**The cost of a wrong answer is real.** Missing a registration deadline or misunderstanding the attendance rule can cost a student an exam or a semester. That makes trust, not speed, the core product requirement.

> **Discovery to validate (before pilot):** survey 20+ students across years and ask
> (a) the last rule-related question they had, (b) where they looked, (c) how long it took,
> (d) whether the answer turned out to be right. Record findings in `docs/discovery.md`
> and update this section with real numbers.

## 2. Users

| User | Job to be done | What they care about |
|---|---|---|
| **First-year student** | "Tell me what I have to do and by when, without having to ask a senior." | Plain language, deadlines, not feeling lost |
| **Senior student** | "Check a specific rule quickly (backlogs, summer courses, branch change, internships)." | Precision, exact clause, speed |
| **Student asking in Hindi/Odia** | "Ask in the language I think in." | Being understood; answer in the same language |
| **Admin / content owner** (student body or office) | "Know what students are confused about and keep the source documents current." | Visibility into gaps, low maintenance |

## 3. Goals and non-goals

**Goals**
1. Give accurate, cited answers to rule and process questions from official documents.
2. Say "I don't know" instead of guessing, and route the student to the right office.
3. Turn every unanswered question into a visible content gap the admin can fix.

**Non-goals (v1)**
- No logins or personal data: it will not answer "what is *my* attendance" or "*my* CGPA". That needs ERP/NITRIS integration and consent.
- No outside knowledge: answers come only from indexed documents.
- Not an official institute service, and it doesn't replace any office.

## 4. Success metrics

| Type | Metric | v1 target | How measured |
|---|---|---|---|
| **North star** | Correctly answered questions per week | Grows week on week during pilot | `answered` status × helpful rating |
| Quality | Answer correctness on golden set | ≥ 90% | `eval/run_eval.py` (LLM judge vs human reference) |
| Quality | Retrieval hit rate@6 | ≥ 95% | Eval: expected doc in top-k |
| **Guardrail** | Refusal accuracy (answers when it should, refuses when it should) | ≥ 90% | Eval set includes gaps, out-of-scope, adversarial |
| **Guardrail** | 👎 rate on answered questions | ≤ 10% | `feedback` table |
| Coverage | Answer rate (answered ÷ answered + not found) | ≥ 70% after first gap-filling pass | `interactions` table |
| Experience | p95 latency | ≤ 6 s | Logged per interaction |

The guardrails matter more than the north star: a tool that answers more questions but gets more of them wrong is worse than useless here.

## 5. Requirements

| Pri | Requirement | Status |
|---|---|---|
| P0 | Chat answers grounded only in indexed official documents | ✅ Built |
| P0 | Every answer shows its source document and page, with a link to the official PDF | ✅ Built |
| P0 | Refuses when the answer is not in the documents; model answers with no citation are automatically downgraded to "not found" | ✅ Built |
| P0 | "Who can help" escalation to the relevant office when unanswered | ✅ Built |
| P0 | Out-of-scope and prompt-injection handling | ✅ Built |
| P0 | Offline evaluation harness with golden set (retrieval, correctness, refusal, latency) | ✅ Built |
| P1 | Follow-up questions understood in context ("what about PG students?") | ✅ Built (query rewriting) |
| P1 | Hybrid retrieval: keyword (exact terms like CGPA, clause numbers) + semantic (paraphrases) | ✅ Built |
| P1 | Topic filter (academics, exams, fees, hostel, placements, scholarships) | ✅ Built |
| P1 | Hindi / Odia questions answered in the same language | ✅ Built (multilingual embeddings + prompt) |
| P1 | 👍/👎 feedback with optional comment | ✅ Built |
| P1 | Admin console: usage, answer rate, content gaps, negative feedback, SQL shown per metric | ✅ Built |
| P1 | Admin document upload + re-index | ✅ Built |
| P1 | Suggested follow-up questions | ✅ Built |
| P2 | Document freshness warnings (flag calendars from a past academic year) | Roadmap |
| P2 | WhatsApp / Telegram bot interface (where students already are) | Roadmap |
| P2 | Scheduled scraper for new notices on nitrkl.ac.in | Roadmap |
| P2 | Personal answers via NITRIS integration (with consent) | Roadmap |

## 6. Core flow

```
Student asks → guardrails → (rewrite follow-up) → hybrid search (top 6 chunks)
   → Claude answers in JSON {status, answer, citations, category, follow_ups}
      ├─ answered      → answer + sources + follow-ups + 👍/👎
      ├─ not_found     → honest "not in my documents" + who can help  → logged as content gap
      └─ out_of_scope  → polite redirect
   → every interaction logged to SQLite → admin dashboard
```

## 7. Key decisions and trade-offs

| Decision | Chosen | Alternative considered | Why |
|---|---|---|---|
| Answer source | RAG over official PDFs | Fine-tuning; plain LLM | Rules change yearly: re-indexing a PDF beats retraining, and citations are only possible with retrieval. A plain LLM hallucinates institute-specific rules. |
| Retrieval | Hybrid BM25 + Pinecone with Reciprocal Rank Fusion | Vector-only | Student questions mix exact tokens (CGPA, "Autumn 2026-27") and paraphrases. RRF merges both without score calibration. Keyword-only mode means anyone can run it without a Pinecone key. |
| Embeddings | Google's free embedding endpoint | Local sentence-transformers; paid embeddings | No ML model in the deploy (torch alone would blow the free hosting limit), no cost, and multilingual out of the box. |
| Vector index | Local NumPy file committed to the repo | Hosted vector DB (Pinecone) | At a few thousand chunks, cosine over a matrix is fast and removes an external service. Pinecone sits behind the same interface for when the corpus grows. |
| Output format | Structured JSON with explicit status | Free-text streaming | An explicit `not_found` state is what enables escalation, gap analytics, and eval. Cost: no token streaming; acceptable because answers are short. |
| Model provider | Free tier (Gemini by default), behind a provider interface | Single paid provider | The project had a hard constraint of zero budget, and free tiers change often. One HTTP layer means switching provider is a config line, and the eval set measures what the switch costs in quality. |
| Answer model | Flash-class model, larger model as eval judge | Largest available model everywhere | Short grounded answers don't need the largest model; free-tier rate limits and latency do matter. |
| Uncited answers | Auto-downgraded to "not found" | Trust the model | An answer with no evidence is the most likely hallucination. Losing a few correct answers is cheaper than showing a wrong one. |
| Logging | SQLite | Postgres / analytics SaaS | Zero setup and plain SQL. Swap for Postgres before a campus-wide launch (see risks). |
| Interface | Streamlit web app | WhatsApp bot | Fastest path to a testable product with an admin console; WhatsApp is the P2 distribution bet once quality is proven. |

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Outdated document gives a confidently wrong answer | High | Answers state the regulation year/session; indexed date shown; freshness warnings on roadmap; disclaimer to verify money- and deadline-related rules |
| Hallucinated rule | High | Documents-only prompt, citation requirement with auto-downgrade, golden-set refusal tests, 👎 review queue |
| Students treat it as official | Medium | Clear "student-built, not official" notice; always links to the official source |
| Prompt injection via question or documents | Medium | Documents treated as data in the prompt; pattern-based blocking; adversarial cases in the eval set |
| SQLite logs reset on Streamlit Cloud restarts | Low for pilot | Acceptable for pilot; move to hosted Postgres before wider rollout |
| Free-tier rate limits hit during a demo or a busy day | Medium | Provider layer backs off and retries; keyword-only mode still answers if embeddings quota runs out; provider is swappable in one config line |
| Free-tier prompts may be used by the provider to improve its models | Low | Only public institute documents are indexed and no student accounts exist; documented openly in the README |

## 9. Rollout plan

1. **Internal quality bar:** golden set to 40+ questions with human-written references; hit correctness ≥ 90% and refusal accuracy ≥ 90%.
2. **Pilot (2 weeks):** one batch or one hall of residence, 30-50 students. Watch the 👎 queue daily.
3. **Fix gaps:** add the top 10 missing documents from the content-gaps view; re-run eval.
4. **Decide:** expand campus-wide if helpful ≥ 85% and 👎 ≤ 10%; otherwise iterate on retrieval and coverage first.
5. **Distribution:** WhatsApp/Telegram interface, announced through student bodies at semester registration (peak question volume).

## 10. Open questions

- Who owns document freshness long-term: a student club, or an institute office?
- Would the Academic Section endorse a verified-sources list?
- Is personal data (attendance, grades) worth the consent and security burden, or is general guidance enough?
