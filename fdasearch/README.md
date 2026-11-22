# fdasearch: zero-shot baselines for 510(k) predicate retrieval

Task: given a new (subject) device's Indications for Use + Device Description, rank earlier 510(k)
clearances so that the predicates its summary cites come first. This repo builds the dataset from
public FDA records and evaluates untuned retrieval methods. See `results/REPORT.md` for numbers.

## Pipeline

```
data/raw/device-510k.json.zip      openFDA 510(k) metadata (176,683 records, export 2026-11-18)
data/pdfs/K######.pdf              510(k) summary PDFs (decisions 2010-2026 with statement_or_summary == "Summary")
data/pdfs_closure/K######.pdf      cited predicates outside 2010+ ("closure fetch", 1,021 PDFs; 298 have no PDF at FDA)
data/extracted.jsonl               per-PDF text, sections, cited K-numbers (+ predicate/reference label); image-only PDFs OCR'd
data/extracted_closure.jsonl       same for the closure PDFs
data/corpus.parquet                one row per device with usable text (v2; v1 = 47,531 docs kept in data/v1/)
data/queries.parquet               subjects 2010-2023 (train), 2024 (dev), 2025-2026 (test) with >=1 predicate in corpus
data/predx/                        the same dataset in the team's predx layout (corpus / queries_* / train_pairs .jsonl.gz, dataset_card.json)
results_v1/                        metrics and ranks of the first run (no OCR, 2010+ corpus only, no SE-sentence stripping)
results/metrics.csv, REPORT.md     metrics with bootstrap 95% CIs; per-query ranks in results/ranks/
results/emb/*.npz                  cached document/query embeddings keyed by K-number
```

Scripts (run from `E:\fdasearch`, Python in `.venv`):

1. `scripts/download_pdfs.py --min-year 2010 --workers 6`
   Downloads summary PDFs from accessdata.fda.gov. The server rejects curl's default user agent
   with an "abuse detection" page; the script sends an identifying research UA and gets ~25 PDFs/s.
   URL folder is `pdf` for 1976-2000 and `pdfYY` otherwise. ~0.6% of summaries are 404.
2. `scripts/extract_text.py --workers 8`, then `--ocr --workers 1 --redo-low-text`
   PyMuPDF text extraction; PDFs with <100 chars/page are re-done with RapidOCR on the GPU
   (150 dpi, first 12 pages, ~0.65 s/page; 775 of the 990 image-only 2010+ PDFs became usable). Finds the
   "510(k) Summary" header, then the Indications for Use and Device Description sections by regex
   headers (IFU found for 97%, description for 80%; otherwise the summary body is used).
   Cited K-numbers come from the summary portion; each is labeled "reference" if the words
   "reference device" occur more recently than "predicate" in the preceding 700 characters.
   `scripts/fetch_closure.py` + `extract_text.py --ocr --pdf-dir data/pdfs_closure --out data/extracted_closure.jsonl`
   download and extract the predicates cited by queries that are outside the 2010+ window (the team's
   "closure fetch"), so pre-2010 predicates become retrievable corpus documents.
3. `scripts/build_dataset.py`
   Joins metadata, masks all K-numbers out of the text, drops sentences that mention predicates /
   substantial equivalence / 510(k) (the team's SE_LANGUAGE_IN_INPUT leak), flags queries whose
   predicate's device name appears verbatim in the query text, builds corpus and train/dev/test splits.
   Positives = regex-labeled predicates present in the corpus and cleared before the subject.
   `scripts/export_predx.py` writes the same data in the predx file layout for the team's training code.
4. `scripts/run_baselines.py --pc --methods bm25,tfidf,lsa,pop,emb:<hf model>,ollama:<model>,rrf:a|b`
   Candidates for each query are corpus docs dated before the subject's decision. `--pc` additionally
   evaluates each method restricted to the subject's product code. Embedding inputs are truncated
   to 512 tokens (device name + IFU + start of description). `rrf:` fuses saved top-100 lists.
5. `scripts/summarize_results.py` writes `results/REPORT.md` (overall, by panel, by product-code pool size).
6. `scripts/llm_label_check.py` (run on a Spark with Ollama) asks gemma4:31b for predicate/reference
   K-numbers on 100 random 2024-2026 summaries; output in `results/label_check_llm.jsonl`.

## Results (dataset v2, test split = 2025-2026 subjects, n=4912)

Full table with CIs in `results/REPORT.md`. Recall@10 / Hit@1 / Hit@10 / Hit@100 / MRR, in percent:

| method | Recall@10 | Hit@1 | Hit@10 | Hit@100 | MRR |
|---|---|---|---|---|---|
| RRF(TF-IDF, Gemini embedding-001) | 69.9 | 48.4 | 82.6 | 96.1 | 60.5 |
| TF-IDF | 69.3 | 49.7 | 82.5 | 95.4 | 61.3 |
| RRF(TF-IDF, EmbeddingGemma) | 69.1 | 48.4 | 81.9 | 96.0 | 60.3 |
| RRF(TF-IDF, bge-m3) | 67.6 | 48.1 | 80.6 | 95.4 | 59.7 |
| BM25 | 65.4 | 47.3 | 78.4 | 93.3 | 58.3 |
| Gemini embedding-001 | 65.1 | 45.3 | 77.6 | 94.2 | 56.4 |
| EmbeddingGemma (Ollama) | 62.9 | 45.4 | 75.5 | 93.1 | 56.1 |
| bge-m3 | 61.8 | 45.4 | 74.6 | 91.4 | 55.7 |
| LSA | 56.3 | 38.0 | 68.9 | 89.6 | 48.4 |
| popularity | 19.2 | 5.9 | 25.5 | 69.1 | 12.6 |

Product-code restriction (`+pc`) lowers Recall@10 by 2-4 points for every method because about 10% of
positives sit outside the subject's product code. Every method scores 1.2-2.4 points lower than in v1
(`results_v1/`): v2 counts pre-2010 predicates as positives (closure fetch), adds 934 OCR'd scans, and strips
substantial-equivalence sentences from the texts, so the task got harder rather than the methods worse.
The ranking of methods is unchanged between v1 and v2.

## Data bundle for the Colab (hermetic w.r.t. data)

`bundle/` (670 MB; mirrored to the Drive folder `fdasearch_data_v2`) holds everything the notebook needs so it never
touches FDA servers: the openFDA export, `extracted.jsonl.gz` + `extracted_closure.jsonl.gz` (PyMuPDF/OCR text and
cited K-numbers for every PDF), `corpus.parquet`, `queries.parquet`, `predx.zip` (team layout), the reference
`metrics.csv` / `REPORT.md`, and `manifest.json` with sha256 per file. The notebook verifies the checksums, rebuilds a
one-panel dataset from the extracted text in quick mode, and uses the shipped tables in full mode. Models are still
pulled from Hugging Face / the Gemini API and embedded in Colab. `bundle_emb_cache/` keeps the local embedding caches
of six dense models (1.7 GB) for anyone who wants to skip embedding; drop its files into `results/emb/`.

## Metrics

Recall@10 (fraction of a subject's positives in the top 10), Hit@1/Hit@10/Hit@100 (any positive in
top k), MRR, and the same for the primary predicate only (first K-number cited). CIs are bootstrap
over queries (1000 resamples).

## Known label noise

On the 100-summary LLM check, the first-cited K-number agrees with the LLM's predicate list in
91% of summaries; across all cited K-numbers the regex labels have precision 0.70 / recall 0.93
against the LLM (over-inclusion is mostly reference devices and the subject's own prior clearances).
All scores therefore underestimate usefulness; primary-only metrics are the cleanest.

## Relation to the team's predx pipeline

`predx_colab_test_2.ipynb` (team notebook) tests a package `predx` shipped separately as `predicate-data.zip`;
the package itself is not in this repo. Checks against the notebook: on the 7 PDFs with hand-verified primary
predicates, this extractor agrees on all 3 digital PDFs and, after adding OCR, on all 4 scanned 1990s PDFs.
Borrowed from their design: OCR fallback, closure fetch of cited predicates, SE-language stripping and
name-leak flags, train split + pairs, and the predx dataset layout.

## Environment

Local: RTX 4090, `.venv` with CUDA torch (2.11+cu128), sentence-transformers, bm25s, rapidocr-onnxruntime with
onnxruntime-gpu 1.22 (import torch before RapidOCR so the CUDA DLLs load). Gemini embeddings need `GEMINI_API_KEY` in `.env`. EmbeddingGemma is served from
Ollama (`embeddinggemma`) because the Hugging Face weights are gated. nomic-embed-text also via Ollama
(its HF remote code fails with transformers 4.57). Sparks (`ssh spark`, `ssh spark2`) have Ollama but no torch.
