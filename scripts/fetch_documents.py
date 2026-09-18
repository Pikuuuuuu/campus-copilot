"""Download a starter set of official NIT Rourkela documents into data/raw/.

Only public documents from nitrkl.ac.in are listed. Before you rely on any of them,
open the Academic Regulations / Academic Calendar pages on nitrkl.ac.in and replace
these URLs with the most recent versions (regulations and calendars change yearly).

Usage: python scripts/fetch_documents.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DOCUMENTS = [
    {
        "category": "academics",
        "file": "ug_academic_regulations.pdf",
        "title": "UG Academic Regulations (B.Tech / Dual Degree / B.Arch / Int. M.Sc.)",
        "url": "https://www.nitrkl.ac.in/docs/AcademicRegulation/04012020192115698.pdf",
    },
    {
        "category": "academics",
        "file": "academic_calendar_2026_27.pdf",
        "title": "Academic Calendar 2026-27",
        "url": "https://www.nitrkl.ac.in/docs/AcademicCalendar/2026-27/06072026131143827.pdf",
    },
    {
        "category": "general",
        "file": "first_year_ug_instructions.pdf",
        "title": "Instructions for 1st Year UG Students (Notice)",
        "url": "https://nitrkl.ac.in/docs/Announcement/03122021153353049.pdf",
    },
    # Add hostel rules, fee notices, scholarship notices and CDC placement policy here.
]


def main() -> int:
    manifest_path = RAW / "sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = 0
    for doc in DOCUMENTS:
        target = RAW / doc["category"] / doc["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(doc["url"], headers={"User-Agent": "Mozilla/5.0 CampusCopilot"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                target.write_bytes(resp.read())
            manifest[doc["file"]] = {"title": doc["title"], "url": doc["url"]}
            print(f"✓ {doc['title']}")
        except Exception as exc:
            failures += 1
            print(f"✗ {doc['title']}: {exc}  (download it manually from {doc['url']})")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nNext: python scripts/ingest.py")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
