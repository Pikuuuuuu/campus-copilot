# Source documents

Put official documents in a folder named after their topic:

```
data/raw/academics/ug_academic_regulations.pdf
data/raw/hostel/hall_rules_2026.pdf
data/raw/fees/fee_notice_autumn_2026.pdf
```

Topics: academics, exams, fees, hostel, placements, scholarships, general.

`sources.json` maps each file name to a display title and its official URL, so citations
link back to nitrkl.ac.in. `python scripts/fetch_documents.py` downloads a starter set and
fills this in; `python scripts/ingest.py` rebuilds the index.

Only add documents that are publicly available. Never add anything with student personal data.
