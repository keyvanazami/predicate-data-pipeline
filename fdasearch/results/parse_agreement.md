# Parse agreement: predx parser vs fdasearch extraction

Run 2026-11-21 with `scripts/parse_agreement.py` (predx at commit e0295aa of schukka28/predicate-data-pipeline).
Sample: 3,089 documents = the 100 LLM-checked summaries + 3,000 random 2010+ digital (non-OCR) corpus documents.
Both parsers read the same PyMuPDF page text (first 40 pages). Per-document rows in `parse_agreement.csv`.

| Measure | fdasearch | predx |
|---|---|---|
| primary predicate found | 91.4% | 93.4% (3.3% via first-other fallback; 31.3% flagged ambiguous) |
| same primary when both found (91.4% of docs) | 95.9% | |
| gold set (primary + additional) exact match | 81.7% (mean Jaccard 0.888) | |
| mean gold-set size | 1.94 | 1.74 |
| IFU found | 96.9% | 96.2% (form page 2,245, summary 728) |
| IFU overlap when both found | median Jaccard 0.72, median containment 1.00 | |
| description found | 80.5% | 94.8% (heading 2,330, summary body 599) |
| description overlap when both found | median Jaccard 0.98, median containment 1.00 | |
| gold precision / recall vs LLM labels (100 docs) | 0.70 / 0.93 | 0.71 / 0.95 |
| primary in LLM predicate set (90 docs) | 93.3% | 95.6% |

Reading: the two parsers agree on the primary predicate in 96% of documents and differ mostly on long predicate
lists, where fdasearch keeps every K-number in the predicate section and predx keeps fewer. predx is slightly
closer to the LLM labels on both precision and recall and recovers a description for 14 points more documents
(its summary-body fallback happens at parse time; fdasearch's happens at dataset build). The IFU texts overlap by
containment, so the difference there is section boundaries and caps, not content. Not covered by this sample:
OCR'd scans and pre-2010 closure documents.

Conclusion: adopt the predx parser as the single parser; keep fdasearch's PyMuPDF + RapidOCR page text as its input.
