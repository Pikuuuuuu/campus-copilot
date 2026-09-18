"""Offline evaluation of Campus Copilot against a hand-labelled golden set.

Metrics
- retrieval_hit_rate : % of answerable questions where the expected document is in the top-k results
- answer_correctness : % of answerable questions answered AND judged correct vs the reference answer
- refusal_accuracy   : % of all questions where "answered vs refused" matched what should happen
- latency p50 / p95

Rows whose expected_answer starts with "TODO" still count for retrieval and refusal,
but are skipped for correctness until you write the reference answer.

Usage: python eval/run_eval.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, llm  # noqa: E402
from core.ingest import load_chunks  # noqa: E402
from core.rag import CampusCopilot  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows_in = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows_in = rows_in[: args.limit]

    chunks, _ = load_chunks()
    if not chunks:
        sys.exit("No index found. Run scripts/fetch_documents.py and scripts/ingest.py first.")
    copilot = CampusCopilot(chunks, log=False)

    rows, latencies = [], []
    retrieval_hits = retrieval_total = 0
    correct = correct_total = 0
    refusal_ok = 0

    for item in rows_in:
        q = item["question"]
        should_answer = item["should_answer"]

        hit_sources = [h.chunk.source for h in copilot.retriever.search(q, k=config.TOP_K)]
        retrieved = None
        if should_answer and item.get("expected_source"):
            retrieval_total += 1
            retrieved = item["expected_source"] in hit_sources
            retrieval_hits += int(retrieved)

        result = copilot.ask(q)
        latencies.append(result.latency_ms)
        answered = result.status == "answered"
        refusal_ok += int(answered == should_answer)

        verdict = None
        reference = item.get("expected_answer", "")
        if should_answer and reference and not reference.startswith("TODO"):
            correct_total += 1
            if answered:
                verdict = llm.judge(q, reference, result.answer)
                correct += int(verdict["correct"])
            else:
                verdict = {"correct": False, "reason": f"assistant returned {result.status}"}

        rows.append({
            "id": item["id"],
            "question": q,
            "should_answer": should_answer,
            "status": result.status,
            "retrieved_expected_doc": retrieved,
            "correct": None if verdict is None else verdict["correct"],
            "judge_reason": None if verdict is None else verdict["reason"],
            "cited": ", ".join(f"{c.source} p{c.page}" for c in result.sources),
            "latency_ms": result.latency_ms,
        })
        mark = "✓" if answered == should_answer else "✗"
        print(f"{mark} {item['id']:<10} {result.status:<13} {result.latency_ms:>5} ms  {q[:60]}")

    metrics = {
        "retrieval_hit_rate": pct(retrieval_hits, retrieval_total),
        "answer_correctness": pct(correct, correct_total),
        "refusal_accuracy": pct(refusal_ok, len(rows)),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": percentile(latencies, 0.95),
        "graded_for_correctness": correct_total,
    }
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(rows),
        "provider": config.LLM_PROVIDER,
        "answer_model": config.ANSWER_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "retrieval_mode": copilot.retriever.mode,
        "top_k": config.TOP_K,
        "chunk_chars": config.CHUNK_CHARS,
        "metrics": metrics,
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in (f"run-{stamp}.json", "latest.json"):
        (RESULTS / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        f"# Eval run {report['run_at']}",
        f"Provider `{config.LLM_PROVIDER}` · model `{config.ANSWER_MODEL}` · {copilot.retriever.mode} · top_k={config.TOP_K} · chunk={config.CHUNK_CHARS}",
        "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in metrics.items()],
    ]
    (RESULTS / "latest.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md[3:]))


if __name__ == "__main__":
    main()
